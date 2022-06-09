import os 
import cv2 
import numpy as np 
from models import SICE, mse_loss
from utils import load_dataset_sice, split_train_valid, data_generator, write_history, resize, read_history
from utils import datagenerator
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
import tensorflow as tf

def luminanace_train(dataset_dir, batch_size, epochs, weight_dir):
    model = SICE()
    l_network = model.luminance_enhancement_network()
    l_network.summary() 
    l_network.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9), loss=mse_loss)
    x_data, y_data = load_dataset_sice(dataset_dir=dataset_dir)  
    x_train, y_train, x_valid, y_valid = split_train_valid(x_data, y_data, valid_ratio=0.2)
    train_datagen = data_generator(x_train, y_train, batch_size=batch_size, crop_size=129, mode='base')  
    valid_datagen = data_generator(x_valid, y_valid, batch_size=batch_size//3, crop_size=129, mode='base')  

    steps_per_epoch = len(x_train) // batch_size
    valid_steps = int(steps_per_epoch//3+1)
    train(train_datagen, valid_datagen, epochs, steps_per_epoch, valid_steps, weight_dir)


def direct_train(epochs, train_data_dir, valid_data_dir, batch_size, weight_dir, gpu_id=0, last_weight_file=None):
    with tf.device(f'/device:GPU:{gpu_id}'):
        model = SICE()   
        network = model.direct_network()
        network.summary() 
        network.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9), loss=mse_loss)
        train_datagen = datagenerator(train_data_dir, batch_size, img_size=320)
        valid_datagen = datagenerator(valid_data_dir, batch_size, img_size=320)

        num_train_imgs = len(os.listdir(train_data_dir))
        num_valid_imgs = len(os.listdir(valid_data_dir))
        steps_per_epoch = num_train_imgs // batch_size
        valid_steps = num_valid_imgs // batch_size
        print('number train images:', num_train_imgs)
        print('number valid images:', num_train_imgs)
        print('steps per epoch:', steps_per_epoch)
        print('valid steps: ', valid_steps)
        train(network, train_datagen, valid_datagen, epochs, steps_per_epoch, valid_steps, weight_dir, last_weight_file=last_weight_file)

def train(model, train_datagen, valid_datagen, epochs, steps_per_epoch, valid_steps, weight_dir, decay=0.0001, last_weight_file=None):
    
    os.makedirs(weight_dir, exist_ok=True)
    
    history = [['epochs', 'train loss', 'valid loss', 'learning rate']]
    if last_weight_file == None:
        lr = 0.1     
        last_weight_file = ''
        min_valid_loss = 10000
        start_epoch = 0
    else:
        # read history:
        last_epoch = int(last_weight_file.split('_')[1]) 
        last_weight_file = os.path.join(weight_dir, last_weight_file)
        model.load_weights(last_weight_file)
        train_loss, valid_loss, learning_rate = read_history(os.path.join(weight_dir, 'history.txt'))   
        for i in range(last_epoch + 1):
            history.append([str(i), str(train_loss[i]), str(valid_loss[i]), str(learning_rate[i])])        
        lr = learning_rate[last_epoch]
        min_valid_loss = valid_loss[last_epoch]
        start_epoch = last_epoch + 1        
        print(f'last epoch: {last_epoch}, min_valid_loss: {min_valid_loss}, learning_rate: {lr}')

    for epoch in range(start_epoch, epochs):    
        # 30 epoch 마다 1/10로 줄임 
        if epoch > 0 and epoch % 30 == 0:
            lr = lr/10.
        else:
            # 매 epochs 마다 아래와 같이 줄임
            lr = lr /(1 + decay * epoch)
        K.set_value(model.optimizer.learning_rate, lr)

        # training 
        model.trainable = True 
        m_train_loss = 0
        for step in range(steps_per_epoch):
            train_data = next(train_datagen)
            train_loss = model.train_on_batch(train_data[0], train_data[1])
            m_train_loss += train_loss
            print(f'epoch {epoch}/{epochs} - step {step}/{steps_per_epoch} : loss = {train_loss:.4f}')
        m_train_loss = m_train_loss/steps_per_epoch
        
        # validation 
        model.trainable = False
        m_valid_loss = 0
        for _ in range(valid_steps):
            valid_data = next(valid_datagen)
            m_valid_loss += model.evaluate(valid_data[0], valid_data[1])
        m_valid_loss = m_valid_loss/valid_steps
        print(f'epoch {epoch}/{epochs} : loss = {m_train_loss:.4f}, valid loss = {m_valid_loss:.4f}')

        # best weight 
        if m_valid_loss < min_valid_loss:
            min_valid_loss = m_valid_loss
            weight_file = f'model_{epoch}_{m_valid_loss:.3f}.h5'            
            model.save_weights(os.path.join(weight_dir, weight_file))
            
            # test image using best weigths
            img = resize('D:\\projects\\_datasets\\SICE\\Dataset_Part1\\68\\3.JPG', r=1)
            label = resize('D:\\projects\\_datasets\\SICE\\Dataset_Part1\\Label\\68.JPG', r=1)        
            predict = model.predict(np.expand_dims(img, axis=0))
            cv2.imwrite(os.path.join(weight_dir, f'{epoch}.png'), np.hstack([img * 255, label*255, predict[0]*255]))
                
        # last weight    
        # 이전의 weight 삭제 
        if os.path.isfile(last_weight_file):
            os.remove(last_weight_file)        
        # 새로운 weight 기록
        last_weight_file = os.path.join(weight_dir, f'model_{epoch}_last.h5')        
        model.save_weights(last_weight_file)

        # history 
        history.append([str(epoch), 
                        str(f'{m_train_loss:.7f}'), 
                        str(f'{m_valid_loss:.7f}'), 
                        str(K.get_value(model.optimizer.learning_rate))])
        history_save_file = os.path.join(weight_dir, 'history.txt')             
        write_history(history, history_save_file)
    print('finished!!')

if __name__=='__main__':
    epochs = 300
    batch_size = 10
    train_data_dir = './dataset/train_5_2'
    valid_data_dir = './dataset/valid_5_2'
    weight_dir = './weights/train_5_2_4'
    last_weight_file = None
    direct_train(epochs=epochs, batch_size=batch_size,
                train_data_dir=train_data_dir,
                valid_data_dir=valid_data_dir, 
                weight_dir=weight_dir, 
                gpu_id=1, 
                last_weight_file=last_weight_file)
   