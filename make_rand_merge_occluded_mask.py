import os
import cv2
import torch

data_root = './dataset/Inshop_Random_Merge_Occluded/rand_merge_occluded_img/'
dirs = os.listdir(data_root)
new_data_root = './dataset/Inshop_Random_Merge_Occluded/rand_merge_occluded_mask/'
torch.manual_seed(1000)

img_path_list = ['./dataset/Inshop_50_Center_Black_Square/Inshop_50_Center_Black_Square_img/',
             './dataset/80_100_140_160_Rand_Occluded/80_100_140_160_rand_occluded_img/',
             './dataset/3_COCO_Object_Occluded_img_black/coco_object_occluded_black_img/',
             './dataset/4_COCO_Object_Occluded_img_white/coco_object_occluded_white_img/',
             './dataset/Inshop_Handoccluded/hand_occluded_img/',
             './dataset/Inshop_Objectoccluded/object_occluded_img/']

ann_path_list = ['./dataset/Inshop_50_Center_Black_Square/Inshop_50_Center_Black_Square_mask/',
             './dataset/80_100_140_160_Rand_Occluded/80_100_140_160_rand_occluded_mask/',
             './dataset/4_COCO_Object_Occluded_img_white/mask/',
             './dataset/4_COCO_Object_Occluded_img_white/mask/',
             './dataset/Inshop_Handoccluded/hand_occluded_mask/',
             './dataset/Inshop_Objectoccluded/object_occluded_mask/']

for dir in dirs:
    sex = os.path.join(data_root, dir)
    new_sex = os.path.join(new_data_root, dir)
    if not os.path.exists(new_sex):
        os.makedirs(new_sex)
    if os.path.isdir(sex):
        category = os.listdir(sex)
        for cat in category:
            id = os.path.join(sex, cat)
            new_category = os.path.join(new_sex, cat)
            if not os.path.exists(new_category):
                os.makedirs(new_category)
            if os.path.isdir(id):
                ids = os.listdir(id)
                for i in ids:
                    img = os.path.join(id, i)
                    new_ids = os.path.join(new_category, i)
                    if not os.path.exists(new_ids):
                        os.makedirs(new_ids)
                    if os.path.isdir(img):
                        imgs = os.listdir(img)
                        for name in imgs:
                            data_num = torch.randint(0, 6, (1,)).item()
                            cur_root = ann_path_list[data_num]
                            img_name = os.path.join(cur_root, dir, cat, i, name)
                            read_img = cv2.imread(img_name) # (256, 256, 3)
                            # cv2.imshow('img', read_img)
                            # cv2.waitKey()
                            # cv2.destroyallWindows()

                            new_img_name = os.path.join(new_ids, name)
                            cv2.imwrite(new_img_name, read_img)
