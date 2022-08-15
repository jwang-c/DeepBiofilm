#########################################################################################
# 3D cell segmentation using V-net/3D u-net in the following paper:
#
# J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
# "3D GAN image aynthesis and dataset quality assessment for bacterial biofilm", 2022
#
# Jie Wang, VIVA lab
# Last update: Apr. 17, 2022
#########################################################################################

import os
#from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import tifffile
import torch

class BiofilmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        mask_path = mask_path.replace("fake", "real", 1)
        #print(img_path)
        #print(mask_path)
        image_i = tifffile.imread(img_path)/255.0
        mask_i = tifffile.imread(mask_path)>0

        #print(B_array.shape)
        #print(np.mean(B_array)) 
        # adding the channel info, the input images are grayscale
        [D_A,H_A,W_A] = image_i.shape
        image_r = image_i.reshape(1,D_A,H_A,W_A)
        [D_B,H_B,W_B] = mask_i.shape
        mask_r = mask_i.reshape(1,D_B,H_B,W_B)
        
        # transfrom to tensor
        image = torch.from_numpy(image_r[:,0:8,:,:])
        image = image.type('torch.FloatTensor') # for converting to double tensor (cpu)
        mask = torch.from_numpy(mask_r[:,0:8,:,:])
        mask = mask.type('torch.FloatTensor')

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask