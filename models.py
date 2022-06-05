import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, PReLU, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils import load_dataset_sice, data_generator, split_train_valid
import pandas as pd 
import os 
import cv2 
import numpy as np 

## loss functions
def ssim_loss(y_true, y_pred):
    return tf.reduce_mean((1.0 - tf.image.ssim(y_true, y_pred, 1.0))/2.0)

def l1_loss(y_true, y_pred):
    return tf.losses.mean_absolute_error(y_true - y_pred)

def mse_loss(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true, y_pred)

## convolution layers
def conv_prelu(x, filters, kernel_size, padding='valid', strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(x)
    # x = PReLU()(x)
    return x 

def deconv_prelu(x, filters, kernel_size, padding='valid', strides=1):
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(x)
    # x = PReLU()(x)
    return x 

def conv_prelu_bn(x, filters, kernel_size, padding='valid', strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(x)   
    # x = PReLU()(x)
    x = BatchNormalization()(x)
    return x 

def conv_output(x):
    x = Conv2D(3, (1,1), strides=1)(x)
    return x

class SICE():
    def __init__(self):
        self.width = None
        self.height = None
        
    def callbacks(self, save_weight_dir, weight_name):
        weight_path = os.path.join(save_weight_dir, f'{weight_name}_*epoch*.h5')
        weight_path = weight_path.replace('*epoch*','{epoch:04d}_{val_loss:0.2f}')
        
        callbacks_list = [ModelCheckpoint(filepath=weight_path,
                            monitor='val_loss',
                            save_best_only=True,
                            save_weight_only=True,
                            ), 
                            ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.1,
                                patience=30,                            
                            ),
                            ]
        return callbacks_list

    def luminance_enhancement_network(self):
        input = Input(shape=(self.width, self.height, 3))
        d0 = conv_prelu(input, filters=64, kernel_size=(9,9), strides=2)
        d1 = conv_prelu(d0, filters=64, kernel_size=(5,5), strides=2)
        d2 = conv_prelu(d1, filters=64, kernel_size=(3,3), strides=1)
        u0 = deconv_prelu(d2, filters=64, kernel_size=(3,3), strides=1)
        u1 = deconv_prelu(u0, filters=64, kernel_size=(5,5), strides=2)
        s0 = add([d0, u1])
        u2 = deconv_prelu(s0, filters=3, kernel_size=(9,9), strides=2)
        s1 = add([input, u2])
        output = conv_output(s1)
        return Model(input, output)

    def detail_enhancement_network(self):
        input = Input(shape=(self.width, self.height, 3))
        for i in range(6):
            if i == 0:
                x = conv_prelu(input, filters=64, kernel_size=(3,3), padding='same', strides=1)
            else:            
                x = conv_prelu(x, filters=64, kernel_size=(3,3), padding='same', strides=1)
        x = conv_output(x)
        output = add([input, x])
        return Model(input, output)

    def whole_image_enhancement_network(self):
        input = Input(shape=(self.width, self.height, 3))
        for i in range(6):
            if i == 0:
                x = conv_prelu_bn(input, filters=64, kernel_size=(3,3), padding='same', strides=1)
            else:            
                x = conv_prelu_bn(x, filters=64, kernel_size=(3,3), padding='same', strides=1)
        x = conv_output(x)
        output = add([input, x])
        return Model(input, output)
    
# train 
def luminanace_train2(dataset_dir, batch_size, epochs):
    model = SICE()
    l_network = model.luminance_enhancement_network()
    l_network.summary() 
    l_network.compile(optimizer=Adam(learning_rate=0.1), loss=mse_loss, metrics=['acc'])
    x_data, y_data = load_dataset_sice(dataset_dir=dataset_dir)  
    x_train, y_train, x_valid, y_valid = split_train_valid(x_data, y_data, valid_ratio=0.2)
    train_datagen = data_generator(x_train, y_train, batch_size=batch_size, crop_size=129, mode='base')  
    valid_datagen = data_generator(x_valid, y_valid, batch_size=batch_size, crop_size=129, mode='base')  

    history = l_network.fit(train_datagen,
                    validation_data=valid_datagen, 
                    validation_steps=20, 
                    steps_per_epoch=100, 
                    epochs=epochs,
                    callbacks=model.callbacks('./weights','luminance'))
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    # or save to csv: 
    hist_csv_file = os.path.join('./weights', 'luminance_history.csv')
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

def luminanace_train(dataset_dir, batch_size, epochs):
    model = SICE()
    l_network = model.luminance_enhancement_network()
    l_network.summary() 
    l_network.compile(optimizer=Adam(), loss=mse_loss)
    x_data, y_data = load_dataset_sice(dataset_dir=dataset_dir)  
    x_train, y_train, x_valid, y_valid = split_train_valid(x_data, y_data, valid_ratio=0.2)
    train_datagen = data_generator(x_train, y_train, batch_size=batch_size, crop_size=129, mode='base')  
    valid_datagen = data_generator(x_valid, y_valid, batch_size=batch_size//3, crop_size=129, mode='base')  

    steps_per_epoch = len(x_train) // batch_size
    validation_steps = int(steps_per_epoch//3+1)

    min_valid_loss = 1000000
    for epoch in range(epochs):
        # training process
        l_network.trainable=True
        for step in range(steps_per_epoch):
            x_train_batch, y_train_batch = next(train_datagen)
            train_loss = l_network.train_on_batch( x_train_batch, y_train_batch)
            print(f'Epoch {epoch+1}/{epochs} - step {step+1}/{steps_per_epoch} : train loss = {train_loss:.5f}')
        
        # validation process
        l_network.trainable=False 
        valid_loss = 0
        for _ in range(validation_steps):
            x_valid_batch, y_valid_batch = next(valid_datagen)
            valid_loss += l_network.train_on_batch(x_valid_batch, y_valid_batch)
        valid_loss = valid_loss/validation_steps
        print(f'Epoch {epoch+1}/{epochs} - train loss = {train_loss:.5f}, validation loss = {valid_loss:.5f}')

        if valid_loss <= min_valid_loss:
            min_valid_loss = valid_loss
            # save weight
            l_network.save_weights(f'./weights/lunimnance_{epoch}_{min_valid_loss:.3f}.h5')

            test_image = load_test_image('D:\\projects\\_datasets\\SICE\\Dataset_Part1\\1\\1.jpg')
            teat_label = load_test_image('D:\\projects\\_datasets\\SICE\\Dataset_Part1\\Label\\1.jpg')

            predicted_images = l_network.predict(np.expand_dims(test_image, axis=0))

            for i, predicted_image in enumerate(predicted_images):
                cv2.imwrite(f'{epoch}_{i}.png', np.hstack([test_image*255,                                                          
                                                           teat_label*255,
                                                           predicted_image*255]))
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
    dataset_dir = 'D:\\projects\\_datasets\\SICE\\Dataset_Part1'
    luminanace_train(dataset_dir=dataset_dir, batch_size=80, epochs=1000)
    # test2()

  