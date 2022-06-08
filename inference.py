import cv2 
import numpy as np 
from models import SICE
from utils import resize
import tensorflow as tf 


def inference_direct(weight, gpu_id, img_file, label_file=None, resize_sacle=3, save_file='test.png'):
    with tf.device(f'/device:GPU:{gpu_id}'):
        model = SICE()   
        network = model.direct_network()
        network.load_weights(weight)

        # test image using best weigths
        img = resize(img_file, r=resize_sacle)
        label = resize(label_file, r=resize_sacle)        
        predict = network.predict(np.expand_dims(img, axis=0))
        if label_file == None:
            cv2.imwrite(save_file, np.hstack([img * 255, label*255, np.clip(predict[0]*255,0,255)]))
        else:
            cv2.imwrite(save_file, np.hstack([img * 255, np.clip(predict[0]*255,0,255)]))



if __name__=='__main__':
    weight = './weights/train_6/model_23_0.024.h5'
    img_file = 'D:\\projects\\_datasets\\SICE\\Dataset_Part1\\68\\3.JPG'
    label_file = 'D:\\projects\\_datasets\\SICE\\Dataset_Part1\\Label\\68.JPG'
    inference_direct(weight=weight, gpu_id=1, img_file=img_file, label_file=label_file)


