#########################################################################################
# 3D data loader: input type tif
# Data are unpaired, A for label set, B for real image set
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
#import numpy as np
import torch
#from skimage import io
#import math
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

class Unaligned3d(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        #self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # input images are 3d tiff stacks
        A_array = tifffile.imread(A_path)/255.0
        B_array = tifffile.imread(B_path)/255.0

        # adding the channel info, the input images are grayscale
        [D_A,H_A,W_A] = A_array.shape
        A_img = A_array.reshape(1,D_A,H_A,W_A)
        [D_B,H_B,W_B] = B_array.shape
        B_img = B_array.reshape(1,D_B,H_B,W_B)
        
        # transfrom to tensor
        A = torch.from_numpy(A_img[:,0:8,:,:])
        A = A.type('torch.FloatTensor') # for converting to double tensor (cpu)
        B = torch.from_numpy(B_img[:,0:8,:,:])
        B = B.type('torch.FloatTensor')
        #print(index)
        #print(A.shape)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc


        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'Unaligned3d'
