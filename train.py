import os 
import cv2 
import numpy as np 
from models import SICE, mse_loss
from utils import load_dataset_sice, split_train_valid, data_generator, write_history, resize
from utils import datagenerator
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K


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


def direct_train(epochs, batch_size, weight_dir):
    model = SICE()
    network = model.direct_network()
    network.summary() 
    network.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9), loss=mse_loss)
    train_datagen = datagenerator('./dataset/train', batch_size)
    valid_datagen = datagenerator('./dataset/valid', batch_size)

    steps_per_epoch = 21
    valid_steps = 45
    train(train_datagen, valid_datagen, epochs, steps_per_epoch, valid_steps, weight_dir)


def train(model, train_datagen, valid_datagen, epochs, steps_per_epoch, valid_steps, weight_dir, decay=0.0001):
    
    history = [['epochs', 'train loss', 'valid loss', 'learning rate']]
    last_weight_file = ''

    for epoch in range(epochs):    
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
            m_valid_loss += model.train_on_batch(valid_data[0], valid_data[1])
        m_valid_loss = m_valid_loss/valid_steps
        print(f'epoch {epoch}/{epochs} : loss = {m_train_loss:.4f}, valid loss = {m_valid_loss:.4f}')

        # history 
        history.append([str(epoch), 
                        str(f'{m_train_loss:.4f}'), 
                        str(f'{m_valid_loss:.4f}'), 
                        str(K.get_value(model.optimizer.learning_rate))])
        history_save_file = os.path.join(weight_dir, 'history.txt')             
        write_history(history, history_save_file)

        # best weight 
        if m_valid_loss < min_valid_loss:
            min_valid_loss = m_valid_loss
            weight_file = f'model_{epoch}_{m_valid_loss:.3f}.h5'            
            model.save_weights(os.path.join(weight_dir, weight_file))
            
            # test image using best weigths
            img = resize('dataset\\Dataset_Part1\\68\\2.JPG', r=3)
            label = resize('dataset\\Dataset_Part1\\Label\\68.JPG', r=3)        
            predict = model.predict(np.expand_dims(img, axis=0))
            cv2.imwrite(os.path.joing(weight_dir, f'{epoch}.png'), np.hstack([img * 255, label*255, predict[0]*255]))
                
        # last weight    
        # 이전의 weight 삭제 
        if os.path.isfile(last_weight_file):
            os.remove(last_weight_file)        
        # 새로운 weight 기록
        last_weight_file = os.path.join(weight_dir, f'model_{epoch}_last.h5')        
        model.save_weights(last_weight_file)

    print('finished!!')

def load_test_image(filename):
    img = cv2.imread(filename)
    rows, cols = img.shape[:2]
    img = cv2.resize(img, dsize=(cols//3, rows//3))
    # 네트워크 구조 상 아래 형태의 크기를 가짐 
    rows, cols = img.shape[:2]
    rows = rows - (rows -1) % 4
    cols = cols - (cols -1) % 4
    img = img[:rows, :cols,:]
    return img.astype('float32')/255.0

def detail_train():
    pass 

def whole_train():
    pass 

def test():
    import cv2
    import numpy as np  
    model = SICE()       
    l_network = model.luminance_enhancement_network()
    l_network.load_weights('./weights/luminance_0146_0.03.h5')
    img = cv2.imread('D:\\projects\\_datasets\\SICE\\Dataset_Part1\\2\\3.jpg')
    img = img.astype('float32')/255.0
    rows, cols = img.shape[:2]
    rows_r = rows//129
    cols_r = cols//129
    imgs = [] 
    for y in range(rows_r):
        for x in range(cols_r):
            imgs.append(img[y*129:y*129+129,x*129:x*129+129,:])       
    predicted_imgs = l_network.predict(np.array(imgs))

def test2():
    import cv2
    import numpy as np  
    from utils import get_base_image
    model = SICE()       
    l_network = model.luminance_enhancement_network()
    l_network.load_weights('./weights/luminance_0146_0.03.h5')
    img = cv2.imread('D:\\projects\\_datasets\\SICE\\Dataset_Part1\\1\\1.jpg')
    img = cv2.resize(img,dsize=(129,129))
    rows, cols = img.shape[:2]

    img = cv2.resize(img, dsize=(cols//3, rows//3))
    rows, cols = img.shape[:2]
    rows = (rows - 17)//4 * 4 + 17
    cols = (cols - 17)//4 * 4 + 17
    img = img[:rows, :cols,:]
   
    img = get_base_image(img)
       
    predicted_imgs = l_network.predict(np.expand_dims(img, axis=0))


    cv2.imwrite('p.png',predicted_imgs[0]*255)
    cv2.imwrite('i.png',img*255)


if __name__=='__main__':
    direct_train(100, 40, 'weights')
   