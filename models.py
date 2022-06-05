import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, PReLU, add
from tensorflow.keras.optimizers import Adam

## loss functions
def ssim_loss(y_true, y_pred):
    return tf.reduce_mean((1.0 - tf.image.ssim(y_true, y_pred, 1.0))/2.0)

def l1_loss(y_true, y_pred):
    return tf.losses.mean_absolute_error(y_true - y_pred)

def mse_loss(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true, y_pred)

## convolution layers
def conv_prelu(x, filters, kernel_size, padding='valid', strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    x = PReLU()(x)
    return x 

def deconv_prelu(x, filters, kernel_size, padding='valid', strides=1):
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    x = PReLU()(x)
    return x 

def conv_prelu_bn(x, filters, kernel_size, padding='valid', strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)       
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x 

def conv_output(x):
    x = Conv2D(3, (1,1), strides=1)(x)
    return x

class SICE():
    def __init__(self):
        self.width = None
        self.height = None
  
    def direct_network(self):
        input = Input(shape=(self.width, self.height, 3))
        x = conv_prelu_bn(input, filters=64, kernel_size=(3,3), padding='same', strides=1)
        for _ in range(13):
            x = conv_prelu_bn(x, filters=64, kernel_size=(3,3), padding='same', strides=1)
        output = conv_output(x)
        return Model(input, output)

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
   

  