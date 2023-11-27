import os
import cv2
import torch
import numpy as np

data_root = '../dataset/Consumer2Shop/img/'
dirs = os.listdir(data_root)
# new_img_root = '../dataset/Consumer2shop_30_Black_Center/img/'
new_img_root = '../dataset/Consumer2shop_Random_Black_Square/consumer2shop_random_black_square_img/'
if not os.path.exists(new_img_root):
    os.makedirs(new_img_root)
# new_mask_root = '../dataset/Consumer2shop_30_Black_Center/mask/'
new_mask_root = '../dataset/Consumer2shop_Random_Black_Square/consumer2shop_random_black_square_mask/'
if not os.path.exists(new_mask_root):
    os.makedirs(new_mask_root)
# torch.manual_seed(998)
torch.manual_seed(11)

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
                            read_img = cv2.imread(img_name)

                            if 'shop' in name: # don't occlude shop image
                                new_img_name = os.path.join(new_ids, name)
                                cv2.imwrite(new_img_name, read_img)
                                continue
                            else:
                                H, W, C = read_img.shape
                                # 30x30 center black boxes
                                # a, b = H//2-15, H//2+15
                                # c, d = W//2-15, W//2+15
                                # read_img[a:b, c:d] = 0
                                # new_img_name = os.path.join(new_ids, name)
                                # cv2.imwrite(new_img_name, read_img)
                                #
                                # mask = np.zeros((H, W))
                                # mask[a:b, c:d] = 255
                                # new_mask_name = os.path.join(new_ids_ann, name)
                                # cv2.imwrite(new_mask_name, mask)

                                # random location black boxes
                                a, b = H // 2 - 30, H // 2 + 30
                                c, d = W // 2 - 30, W // 2 + 30
                                rand1 = torch.randint(H // 2 - 30, H // 2 - 10, (1,))
                                rand2 = torch.randint(H // 2 + 10, H // 2 + 30, (1,))
                                rand3 = torch.randint(W // 2 - 30, W // 2 - 10, (1,))
                                rand4 = torch.randint(W // 2 + 10, W // 2 + 30, (1,))
                                read_img[rand1:rand2, rand3:rand4] = 0
                                new_img_name = os.path.join(new_ids, name)
                                cv2.imwrite(new_img_name, read_img)

                                mask = np.zeros((H, W))
                                mask[rand1:rand2, rand3:rand4] = 255
                                new_mask_name = os.path.join(new_ids_ann, name)
                                cv2.imwrite(new_mask_name, mask)