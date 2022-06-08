import os 
import numpy as np 
import random
import cv2 
from time import sleep
from threading import Thread

def get_data(data_dir):
    data_pair = []
    label_dir = 'Label'
    data_img_dirs = os.listdir(data_dir)
    data_img_dirs.remove(label_dir)
    for img_dir in data_img_dirs:
        img_file_list = os.listdir(os.path.join(data_dir, img_dir))
        
        for img_file in img_file_list:  
            lable_file_list = os.listdir(os.path.join(data_dir, label_dir))
            lable_file = [x for x in lable_file_list if img_dir == os.path.splitext(x)[0]][0]
            img_file_path = os.path.join(data_dir, img_dir, img_file)
            label_file_path = os.path.join(data_dir, label_dir, lable_file)
            data_pair.append((img_file_path, label_file_path))
    return data_pair

def split_data_pair(data_pair, valid_ratio=0.2):
    random.shuffle(data_pair)
    train_data = [] 
    valid_data = [] 
    
    total = len(data_pair)
    
    for i, data in enumerate(data_pair):
        if i < int(total*(1-valid_ratio)):
            train_data.append(data)
        else:
            valid_data.append(data)
    
    return train_data, valid_data

def random_crop(save_dir, index, data_pair, num_crop=30, crop_size=129, resize=None):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(data_pair[0])
    label = cv2.imread(data_pair[1])
    
    # check image pair size
    if img.shape[:2] == label.shape[:2]:
        # resize
        if resize != None:
            rows ,cols = img.shape[:2]
            dz = (int(cols*resize), int(rows*resize))
            img = cv2.resize(img, dsize=dz, interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, dsize=dz, interpolation=cv2.INTER_CUBIC)

        for num in range(num_crop):
            rows ,cols = img.shape[:2]
            y = random.randint(0,rows-crop_size)
            x = random.randint(0,cols-crop_size)
            cropped_img = img[y:y+crop_size, x:x+crop_size,:]
            cropped_label = label[y:y+crop_size, x:x+crop_size,:]
            cropped_data = np.hstack([cropped_img, cropped_label])

            cv2.imwrite(os.path.join(save_dir,f'{index}_{num}.png'),cropped_data)

def sequential_crop(save_dir, index, data_pair, crop_size=320, stride=300, r=3):
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(data_pair[0])
    label = cv2.imread(data_pair[1])

    if img.shape == label.shape:
        rows, cols = img.shape[:2]
        dz = (cols//r, rows//r)
        img = cv2.resize(img, dsize=dz, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, dsize=dz, interpolation=cv2.INTER_CUBIC)

        rows, cols = img.shape[:2]

        num = 0
        for y in range(0, rows - crop_size, stride):
            for x in range(0, cols - crop_size, stride):
                cropped_img = img[y:y+crop_size, x:x+crop_size, :]
                cropped_label = label[y:y+crop_size, x:x+crop_size, :]
                cropped_data = np.hstack([cropped_img, cropped_label])
                cv2.imwrite(os.path.join(save_dir,f'{index}_{num}.png'),cropped_data)
                num += 1

def cropping_using_thread(data, save_dir, crop_func, **crop_func_opt):
    os.makedirs(save_dir, exist_ok=True)
    total = len(data)
    step = 30
    for i in range(0, total, step):
        print(f'{i}/{total}')
        threads = [] 
        for j in range(i, min(total, i + step)):
            kwargs = {'save_dir':save_dir, 'index':j, 'data_pair':data[j]}
            kwargs.update(crop_func_opt)
            thread = Thread(target=crop_func, kwargs=kwargs)
            thread.daemon=True
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        
        sleep(1)



if __name__ == '__main__':
    data1 = 'D:\\projects\\_datasets\\SICE\\Dataset_Part1'
    data2 = 'D:\\projects\\_datasets\\SICE\\Dataset_Part2\\Dataset_Part2'

    data1_pair = get_data(data1)
    data2_pair = get_data(data2)
    random_data_pair = data1_pair + data2_pair
    train_data, valid_data = split_data_pair(random_data_pair, valid_ratio=0.2)

    for index in range(len(train_data)):
        sequential_crop('D:\\projects_test\\sice\\dataset\\train_7', index, train_data[index], crop_size=129, stride=100, r=1)
    for index in range(len(valid_data)):
        sequential_crop('D:\\projects_test\\sice\\dataset\\valid_7', index, valid_data[index], crop_size=129, stride=100, r=1)

