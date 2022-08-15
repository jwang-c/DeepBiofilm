#########################################################################################
# 3D data loader: input type tif
# load data from one folder
#
# J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
# "3D GAN image aynthesis and dataset quality assessment for bacterial biofilm", 2022
#
# Jie Wang, VIVA lab
# Last update: Apr. 17, 2022
#########################################################################################

import os.path
import random
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import math
import tifffile

IMG_EXTENSIONS = ['.tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class SingleDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        #self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_array = tifffile.imread(A_path)
        [D_A,H_A,W_A] = A_array.shape
        A_img = A_array.reshape(1,D_A,H_A,W_A)
        # transfrom to tensor
        A = torch.from_numpy(A_img[:,0:8,:,:]/255)
        A = A.type('torch.FloatTensor')
       
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'SingleImageDataset'
