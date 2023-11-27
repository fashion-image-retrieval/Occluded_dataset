import os
import cv2
import torch
import numpy as np

data_root = '../dataset/256_inshop_img/'
dirs = os.listdir(data_root)
new_img_root = '../dataset/mid_30_30_blackbox_inshop/img/'
new_mask_root = '../dataset/mid_30_30_blackbox_inshop/mask/'
torch.manual_seed(1000)

for dir in dirs:
    sex = os.path.join(data_root, dir)
    new_sex = os.path.join(new_img_root, dir)
    new_sex_ann = os.path.join(new_mask_root, dir)
    if not os.path.exists(new_sex):
        os.makedirs(new_sex)
    if not os.path.exists(new_sex_ann):
        os.makedirs(new_sex_ann)
    if os.path.isdir(sex):
        category = os.listdir(sex)
        for cat in category:
            id = os.path.join(sex, cat)
            new_category = os.path.join(new_sex, cat)
            new_category_ann = os.path.join(new_sex_ann, cat)
            if not os.path.exists(new_category):
                os.makedirs(new_category)
            if not os.path.exists(new_category_ann):
                os.makedirs(new_category_ann)
            if os.path.isdir(id):
                ids = os.listdir(id)
                for i in ids:
                    img = os.path.join(id, i)
                    new_ids = os.path.join(new_category, i)
                    new_ids_ann = os.path.join(new_category_ann, i)
                    if not os.path.exists(new_ids):
                        os.makedirs(new_ids)
                    if not os.path.exists(new_ids_ann):
                        os.makedirs(new_ids_ann)
                    if os.path.isdir(img):
                        imgs = os.listdir(img)
                        for name in imgs:
                            print(name, " done!")
                            img_name = os.path.join(img, name)
                            read_img = cv2.imread(img_name)  # (256, 256, 3)
                            # cv2.imshow('img', read_img)
                            # cv2.waitKey()
                            # cv2.destroyallWindows()

                            # 50x50 center black boxes
                            read_img[110:140, 110:140] = 0
                            # cv2.imshow('img', read_img)
                            # cv2.waitKey()
                            # cv2.destroyallWindows()
                            # print()
                            new_img_name = os.path.join(new_ids, name)
                            cv2.imwrite(new_img_name, read_img)

                            mask = np.zeros((256, 256))
                            mask[110:140, 110:140] = 255
                            new_mask_name = os.path.join(new_ids_ann, name)
                            cv2.imwrite(new_mask_name, mask)

                            # random location black boxes
                            # rand1 = torch.randint(80, 100, (1,))
                            # rand2 = torch.randint(140, 160, (1,))
                            # read_img[rand1:rand2, rand1:rand2] = 0
                            # new_img_name = os.path.join(new_ids, name)
                            # cv2.imwrite(new_img_name, read_img)