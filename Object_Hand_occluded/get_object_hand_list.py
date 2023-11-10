import cv2
import os

##### make Object Image list txt file #####
# object_img_path = './dataset/object_image_sr/'
# img_list = os.listdir(object_img_path)
# f = open('./dataset/object_image_sr/object_list.txt', 'w')

##### make Object Image list txt file #####
hand_img_path = './dataset/11K_Hands/hands_imgs/'
img_list = os.listdir(hand_img_path)
f = open('./dataset/11K_Hands/hand_list.txt', 'w')

for name in img_list:
    f.write(name[:-4]+'\n')
f.close()