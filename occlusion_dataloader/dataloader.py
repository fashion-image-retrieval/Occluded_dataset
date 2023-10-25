# import libraries
import os
from matplotlib import pyplot as plt
import torch
import torchvision
import numpy as np

import numpy as np, os, sys, pandas as pd, csv, copy
import torch
import torchvision
import PIL.Image

import dataset
from dataset.Inshop import Inshop_Dataset

nb_workers = 4
data_root = '/mnt/disk31/user/zhiweny/785/project'
sz_batch = 128


# load train dataset
trn_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True, 
                is_inception = True
            ),
            occlusion=False
            )

dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = sz_batch,
        shuffle = True,
        num_workers = nb_workers,
        drop_last = True,
        pin_memory = True
    )
print('load train dataset successfully')

print("Number of classes    : ", trn_dataset.nb_classes())
print("No. of train images  : ", trn_dataset.__len__())
print("Shape of image       : ", trn_dataset[0][0].shape)

# Visualize a few images in the dataset
# You can write your own code, and you don't need to understand the code
# It is highly recommended that you visualize your data augmentation as sanity check

r, c    = [5, 5]
fig, ax = plt.subplots(r, c, figsize= (5, 5))

k       = 0
dtl     = torch.utils.data.DataLoader(
    dataset     = trn_dataset, 
    batch_size  = sz_batch,
    shuffle     = True,
)

for data in dtl:
    x, y = data

    for i in range(r):
        for j in range(c):
            img = x[k].numpy().transpose(1, 2, 0)
            ax[i, j].imshow(img)
            ax[i, j].axis('off')
            k+=1
    break

del dtl