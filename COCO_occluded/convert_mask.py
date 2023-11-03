import cv2
import pandas as pd
import skimage.io as io
from PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
from pycocotools.coco import COCO


save_path = './select_mask'
os.makedirs(save_path,exist_ok=True)

array_list = []
data_info = np.asarray(pd.read_table('./list_mask.txt', header=0))
for i in range(data_info.shape[0]):
    file_path = data_info[i][0]
    mask =Image.open(file_path)
    
    # new_size = min(mask.size)
    # # Calculate the coordinates for the square crop
    # left = (mask.width - new_size) / 2
    # top = (mask.height - new_size) / 2
    # right = (mask.width + new_size) / 2
    # bottom = (mask.height + new_size) / 2

    # # Crop the image into a square
    # square_image = mask.crop((left, top, right, bottom))

    # # Resize the square image to (256, 256)
    # resized_image = square_image.resize((256, 256))
    # resized_image_array = np.asarray(resized_image)
    
    # result_array = np.where(resized_image_array != 0, 1, resized_image_array)
    # reversed_binary_array = 1-result_array
    # cv2.imwrite(os.path.join(save_path,f'mask{i}.jpg'), reversed_binary_array* 255)
    
    image_array = np.array(mask)
    nonzero_coordinates = np.argwhere(image_array != 0)

    # Calculate the bounding box of the non-zero pixels
    (min_x, min_y), (max_x, max_y) = nonzero_coordinates.min(0), nonzero_coordinates.max(0)
    #print((min_y, min_x), (max_y, max_x))

    # Crop the rectangular region containing non-zero pixels
    cropped_image = image_array[min_x:max_x+1, min_y:max_y+1]
    print(cropped_image.shape)
    if cropped_image.shape[0] >=250 or cropped_image.shape[1] >= 250:
        desired_size = (int(cropped_image.shape[1]/2),int(cropped_image.shape[0]/2))
        cropped_image = cv2.resize(cropped_image, desired_size, interpolation=cv2.INTER_NEAREST)
        print('resized size:',cropped_image.shape)
        
    result_array = np.where(cropped_image != 0, 1, cropped_image)
    reversed_binary_array = 1-result_array
    #print(np.unique(reversed_binary_array* 255))
    #cv2.imwrite(os.path.join(save_path,f'mask{i}.jpg'), reversed_binary_array* 255)
    
    # Save the Pillow Image object as an image
    image = Image.fromarray(reversed_binary_array* 255)
    image.save(os.path.join(save_path,f'mask{i}.jpg'))
    
    array_list.append(reversed_binary_array)
#save binary array as npz
print(len(array_list))
np.savez('./mask_arrays.npz', *array_list)