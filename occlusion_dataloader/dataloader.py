# import libraries
import os
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
data_root = './'
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