import os
from tkinter.tix import Tree
import cv2 
import numpy as np 
import random
import threading


## load data
def load_dataset_sice(dataset_dir):
    # dataset_dir= 'D:\\datasets\\SICE\\Dataset_Part1\\'
    dir_list = os.listdir(dataset_dir)
    dir_list.remove('Label')

    x_data_dir_list = dir_list
    y_data_dir = 'Label'
    # 레이블 폴더 내 파일 목록
    y_data_file_list = os.listdir(os.path.join(dataset_dir, y_data_dir))

    x_data = {}
    y_data = {}

    for dir_name in x_data_dir_list:    
        img_dir_path = os.path.join(dataset_dir, dir_name)
        for file_name in os.listdir(img_dir_path):
            file_path = os.path.join(img_dir_path, file_name)
            if os.path.isfile(file_path):
                x_img_path = file_path
                y_img_path = [x for x in y_data_file_list if os.path.splitext(x)[0] == dir_name][0]
                y_img_path = os.path.join(dataset_dir, y_data_dir, y_img_path)

                x_data[os.path.join(dir_name, file_name)] = x_img_path
                y_data[os.path.join(dir_name, file_name)] = y_img_path
              
        
    return x_data, y_data

def split_train_valid(x_data, y_data, valid_ratio=0.2):
    keys = list(x_data.keys())
    total = len(keys)
    random.shuffle(keys)

    x_train = dict()
    y_train = dict()
    x_valid = dict()
    y_valid = dict()

    for i in range(total):
        key = keys[i]
        if i < total * (1 - valid_ratio):
            x_train[key] = x_data[key]
            y_train[key] = y_data[key]
        else:
            x_valid[key] = x_data[key]
            y_valid[key] = y_data[key]

    return x_train, y_train, x_valid, y_valid

## data generater 
class Worker(threading.Thread):
    def __init__(self, x_img_file, y_img_file, crop_size=129, mode='base'):
        super().__init__()
        self.x_img_file = x_img_file          
        self.y_img_file = y_img_file 
        self.crop_size = crop_size
        self.mode = mode
        self._return = None
        
    def run(self):
        x_input_img = cv2.imread(self.x_img_file)
        y_input_img = cv2.imread(self.y_img_file)

        #resize
        rows, cols = x_input_img.shape[:2]
        dsize = (rows//3, cols//3)
        x_input_img = cv2.resize(x_input_img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        y_input_img = cv2.resize(y_input_img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        cropped_x_img, cropped_y_img = random_crop(x_input_img, y_input_img, self.crop_size)

        if self.mode == 'base':
            x_img = get_base_image(cropped_x_img)
            y_img = get_base_image(cropped_y_img)
        elif self.mode == 'detail':
            x_img = get_detail_image(cropped_x_img)
            y_img = get_detail_image(cropped_y_img)
        
        else:
            x_img = cropped_x_img.astype('float32') / 255.0
            y_img = cropped_y_img.astype('float32') / 255.0
        
        self._return = x_img, y_img

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return
        

def get_base_image(img, r=5, eps=0.5**2):
    # img.dtype = uint8
    base = cv2.ximgproc.guidedFilter(img, img, r, eps * 255 * 255)
    return base.astype('float32')/255.0

def get_detail_image( img, r=5, eps=0.5**2):
    base =  get_base_image(img, r, eps)
    detail = img.astype('float32') / 255.0 - base 
    return detail  

def random_crop(img1, img2, crop_size):
    rows, cols = img1.shape[:2]

    x = np.random.randint(0, cols - crop_size)
    y = np.random.randint(0, rows - crop_size)

    cropped_img1 = img1[y:y+crop_size, x:x+crop_size, :]
    cropped_img2 = img2[y:y+crop_size, x:x+crop_size, :]
    
    return cropped_img1, cropped_img2


def data_generator(x_train, y_train, batch_size=4, crop_size=129, mode='base'):
    assert mode in ['base', 'detail', 'whole'], 'check input mode'

    keys = list(x_train.keys())

    x_batch = np.zeros((batch_size, crop_size, crop_size , 3), dtype='float32')
    y_batch = np.zeros((batch_size, crop_size, crop_size , 3), dtype='float32')
   
    while True:
        # data shuffling               
        batch_keys = np.random.choice(keys, batch_size, replace=False)    
       
        threads = []
        for i, key in enumerate(batch_keys):  
            thread = Worker(x_train[key], y_train[key])
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        for i, thread in enumerate(threads):
            x_img, y_img = thread.join()           
            x_batch[i] = x_img
            y_batch[i] = y_img

        yield x_batch, y_batch

if __name__=='__main__':
    import time     
    x_data, y_data = load_dataset_sice('D:\\projects\\_datasets\\SICE\\Dataset_Part1')
    data_gen = data_generator(x_data, y_data, batch_size=4, crop_size=129, mode='whole')

    while True:
        start = time.time()
        x_imgs, y_imgs = next(data_gen)
        xx = [x for x in x_imgs]
        yy = [y for y in y_imgs]
        print(f'delay: {time.time() - start} s')
        ##
        cv2.namedWindow('x_imgs',0)
        cv2.namedWindow('y_imgs',0)
        cv2.imshow('x_imgs',np.hstack(xx))
        cv2.imshow('y_imgs',np.hstack(yy))
        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
