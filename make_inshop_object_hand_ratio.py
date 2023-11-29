import cv2
import os
import numpy as np
import random
import math

random.seed(66)
target_ratio = 0.05 # 0.05, 0.1, 0.2
locat = 'bottom' # 'center', 'top', 'bottom', 'random'

##### Inshop + Object occlusion #####
data_root = '../dataset/256_inshop_img/'
mask_path = '../dataset/object_mask_x4/'
object_path = '../dataset/object_image_sr/'
object_list_path = '../dataset/object_list.txt'

save_path = '../dataset/Inshop_Ratio5_Bottom_Objectoccluded/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_save_path = '../dataset/Inshop_Ratio5_Bottom_Objectoccluded/inshop_ratio5_bottom_object_img/'
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

mask_save_path = '../dataset/Inshop_Ratio5_Bottom_Objectoccluded/inshop_ratio5_bottom_object_mask/'
if not os.path.exists(mask_save_path):
    os.makedirs(mask_save_path)

##### Inshop + Hand occlusion #####
# save_path = '../dataset/Inshop_Small_Handoccluded/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
#
# img_save_path = '../dataset/Inshop_Small_Handoccluded/small_hand_occluded_img/'
# if not os.path.exists(img_save_path):
#     os.makedirs(img_save_path)
#
# mask_save_path = '../dataset/Inshop_Small_Handoccluded/small_hand_occluded_mask/'
# if not os.path.exists(mask_save_path):
#     os.makedirs(mask_save_path)
#
# data_root = '../dataset/256_inshop_img/'
# mask_path = '../dataset/11K_Hands/hands_masks/'
# object_path = '../dataset/11K_Hands/hands_imgs/'
# object_list_path = '../dataset/11K_Hands/hand_list.txt'

object_list = []
f = open(object_list_path, 'r')
while True:
    line = f.readline()
    if not line: break
    object_list.append(line)

mask_number = len(object_list)

dirs = os.listdir(data_root)
for dir in dirs:
    sex = os.path.join(data_root, dir)
    new_sex = os.path.join(img_save_path, dir)
    if not os.path.exists(new_sex):
        os.makedirs(new_sex)
    new_sex_mask = os.path.join(mask_save_path, dir)
    if not os.path.exists(new_sex_mask):
        os.makedirs(new_sex_mask)
    if os.path.isdir(sex):
        category = os.listdir(sex)
        for cat in category:
            id = os.path.join(sex, cat)
            new_category = os.path.join(new_sex, cat)
            if not os.path.exists(new_category):
                os.makedirs(new_category)
            new_category_mask = os.path.join(new_sex_mask, cat)
            if not os.path.exists(new_category_mask):
                os.makedirs(new_category_mask)

            if os.path.isdir(id):
                ids = os.listdir(id)
                for i in ids:
                    print("img name: ", i)
                    img = os.path.join(id, i)
                    final_path = os.path.join(dir, cat, i)

                    new_ids = os.path.join(new_category, i)
                    if not os.path.exists(new_ids):
                        os.makedirs(new_ids)
                    new_ids_mask = os.path.join(new_category_mask, i)
                    if not os.path.exists(new_ids_mask):
                        os.makedirs(new_ids_mask)

                    if os.path.isdir(img):
                        imgs = os.listdir(img)
                        for name in imgs:
                            real_img_name = name
                            img_name = os.path.join(img, name)
                            read_img = cv2.imread(img_name)

                            mask_id = random.randint(0,mask_number-1)
                            mask_name = object_list[mask_id][:-1] + '.png'  # object
                            # mask_name = object_list[mask_id][:-1]+'.jpg' # hand
                            mask = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_GRAYSCALE)
                            mask = cv2.resize(mask, (256, 256))

                            num_pixel = len(mask[mask==255]) # number of pixel that has 255
                            cur_ratio = num_pixel / (256 * 256)
                            resize_scale = target_ratio / cur_ratio
                            # I think we should root resize_scale
                            resize_scale = math.sqrt(resize_scale)

                            # new_mask_name = os.path.join(new_ids_mask, mask_name)
                            new_mask_name = os.path.join(new_ids_mask, name)

                            object_name = object_list[mask_id][:-1]+'.jpeg' # object
                            # object_name = object_list[mask_id][:-1] + '.jpg'  # hand

                            object = cv2.imread(os.path.join(object_path, object_name))

                            mask_y, mask_x  = mask.shape[0], mask.shape[1]

                            #### mask: resize and reshape at random ####
                            # resize_scale = random.uniform(0.05, 0.1)
                            desired_x = min(256, int(mask_x * resize_scale))
                            desired_y = min(256, int(mask_y * resize_scale))
                            desired_size = (desired_x, desired_y)
                            mask = cv2.resize(mask, desired_size)
                            new_pixel = len(mask[mask == 255])
                            object = cv2.resize(object, desired_size)

                            # set a position at random (but within some meaningful range)
                            canvas = np.zeros((256,256))
                            if locat == 'center':
                                init_x = 128 - (desired_size[0] // 2)
                                init_y = 128 - (desired_size[1] // 2)
                            elif locat == 'top':
                                init_x = 0
                                init_y = max(0, 128 - (desired_size[1] // 2))
                            elif locat == 'bottom':
                                init_x = max(0, 255 - (desired_size[0]))
                                init_y = max(0, 128 - (desired_size[1] // 2))
                            elif locat == 'random':
                                # init_x, init_y = random.randint(10,245), random.randint(60,150) # for 0.05
                                # init_x, init_y = random.randint(10, 245), random.randint(50, 180) # for 0.1
                                init_x, init_y = random.randint(0, 150), random.randint(0, 150)  # for 0.2
                                init_x = min(init_x, max(0, 255-desired_size[0]))
                                init_y = min(init_y, max(0, 255-desired_size[1]))
                            else:
                                print("Wrong location value!")
                                raise NotImplementedError

                            end_row = init_x + desired_size[0] if init_x + desired_size[0] <= 256 else 256
                            end_col = init_y + desired_size[1] if init_y + desired_size[1] <= 256 else 256

                            mask_end_row = 256 - init_x if end_row >= 256 else desired_size[0]
                            mask_end_col = 256 - init_y if end_col >= 256 else desired_size[1]
                            # print("init_x, end_row, init_y, end_col, mask_end_row, mask_end_col: ", init_x, end_row, init_y, end_col, mask_end_row, mask_end_col)
                            canvas[init_x:end_row, init_y:end_col] = mask[:mask_end_row, :mask_end_col] # canvas: (256, 256)
                            new_canvas = np.zeros((256, 256, 3), dtype=np.uint8)
                            new_canvas[init_x:end_row, init_y:end_col,:] = object[:mask_end_row, :mask_end_col,:]

                            # cv2.imshow('new canvas', new_canvas)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            final_canvas = np.zeros((256, 256, 3))
                            final_mask = np.zeros((256, 256))
                            final_img = read_img
                            for i in range(256):
                                for j in range(256):
                                    if canvas[i,j] == 255: #if mask is white
                                        final_img[i,j,:] = new_canvas[i,j,:] # change image pixel into object pixel
                                        # final_canvas[i, j, :] = new_canvas[i, j, :]
                                        final_mask[i, j] = 255
                            # cv2.imshow('final_img', final_img)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            cv2.imwrite(new_mask_name, final_mask)
                            # cv2.imwrite(new_mask_name, canvas)

                            # save new img
                            real_final_path = os.path.join(new_ids, real_img_name)
                            cv2.imwrite(real_final_path, final_img)