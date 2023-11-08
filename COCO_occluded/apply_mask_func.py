import cv2
import pandas as pd
import skimage.io as io
from PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import random

'''
This is the function for applying mask:
You should have all folder created as listed in 'list_eval_partition.txt',
otherwise cv2.imwrite may fail.
change resize_scale, pos, color based on need 
'''

def apply_mask(data_info,mask_info,mask_number,resize, pos, color):
    for i in range(data_info.shape[0]):
        # read in each pic
        file_path = data_info[i,0]
        img =Image.open(file_path)
        #print('PIL read shape',img.size) # (256, 256)
        read_img = cv2.imread(file_path)
        #print('cv2 read shape',read_img.shape) # (256, 256, 3)
        
        # apply mask---
        # pick a mask at random
        mask_id = random.randint(0,mask_number-1)
        mask = mask_info[f'arr_{mask_id}']
        mask_y, mask_x  = mask.shape[0], mask.shape[1]
        
        # resize and reshape at random
        # resize_scale = random.uniform(0.5, 1)
        resize_scale = random.uniform(resize[0],resize[1])

        desired_size = (int(mask_x*resize_scale),
                        int(mask_y*resize_scale))
        # if mask is still larger than (256,256)
        if desired_size[0] >= 256 or desired_size[1]>= 256:
            min_scale = min(256/desired_size[0],256/desired_size[1])
            desired_size = (int(desired_size[0]*min_scale),int(desired_size[1]*min_scale))
        resized_mask = cv2.resize(mask, desired_size, interpolation=cv2.INTER_NEAREST)
        
        # set a position at random (but within some meaningful range)
        canvas = np.ones((256,256))
        # init_x, init_y = random.randint(0,150), random.randint(80,100)
        if pos == 'rand':
            init_x, init_y = random.randint(0,150), random.randint(40,140) # a lager range
        if pos =='ur':
            init_x, init_y = 0, 20
        if pos =='ul':
            init_x, init_y = 0,140
        if pos =='lr':
            init_x, init_y = 150,20
        if pos =='ll':
            init_x, init_y = 150,140
        #print(init_x, init_y)
        #print(resized_mask.shape)
        end_row = init_x + resized_mask.shape[0] if init_x + resized_mask.shape[0] <= 256 else 256
        end_col = init_y + resized_mask.shape[1] if init_y + resized_mask.shape[1] <= 256 else 256
        #print(end_row, end_col)
        mask_end_row = 256-init_x if end_row >= 256 else resized_mask.shape[0]
        mask_end_col = 256-init_y if end_col >= 256 else resized_mask.shape[1]
        #print(mask_end_row, mask_end_col)
        canvas[init_x:end_row, init_y:end_col] = resized_mask[:mask_end_row, :mask_end_col]
        
        # choose one way to mask---------
        # element-wise operation (black)
        if color == 'black':
            new_img = read_img*canvas[:, :, np.newaxis] 
        # element-wise operation (white)
        if color == 'white':
            for i in range(256):
                for j in range(256):
                    if canvas[i,j] == 0: #if mask canvas is black
                        read_img[i,j] = 255 #convert image into white
                    
        
        # save new img
        cv2.imwrite(f'./img_COCOoccluded2/{file_path}', new_img) # black mask
        #cv2.imwrite(f'./img_COCOoccluded2/{file_path}', read_img) # white mask

if __name__ == '__main__':
    random.seed(88) # [66, 88]

    mask_info = np.load('./mask_arrays.npz')
    mask_number = 36
    data_info = np.array(pd.read_table('./list_eval_partition.txt', header=1, delim_whitespace=True))[:,:]
    resize_scale = (1, 1.2)
    pos = 'rand' # choose between rand, ur, ul, lr, ll
    color = 'black'
    apply_mask(data_info,mask_info,mask_number, resize_scale, pos, color)
