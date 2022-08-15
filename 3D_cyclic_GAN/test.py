#########################################################################################
# Testing script for 3D Cyclic GAN l21
#
# The code framework is built on DeepSynth provided by 
# Prof. Edward J. Delp and his laboratory at Purdue University
#
# In our paper, we majorly revise the model to accomendate 3D inputs, 
# change the loss functions, and modify the model parameters. 
#
# J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
# "3D GAN image aynthesis and dataset quality assessment for bacterial biofilm", 2022
#
# Jie Wang, VIVA lab
# Last update: Apr. 17, 2022
#########################################################################################
import time
import os
from options.test_options import TestOptions
from data_loader3d_single import SingleDataset
from models.models import create_model
from tifffile import imsave
import os 
import ntpath
import torch

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = SingleDataset(opt)
dataset= torch.utils.data.DataLoader(
                    data_loader,
                    batch_size=opt.batchSize,
                    shuffle=not opt.serial_batches,
                    num_workers=int(opt.nThreads))
dataset_size = len(data_loader)
print('#Testing images = %d' % dataset_size)

model = create_model(opt)

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    #visuals = model.get_current_visuals()
    [real_A,fake_B] = model.get_numpy_images() 
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    short_path = ntpath.basename(img_path[0])
    print(short_path)
    pathname =os.path.join(opt.results_dir, opt.name)
    name = os.path.join(opt.results_dir, opt.name, os.path.splitext(short_path)[0])
    print(name)
    saveimf_name = '%s_fake.tif' % (name)
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    imsave(saveimf_name,fake_B)
    
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    saveimr_name = '%s_real.tif' % (name)
    imsave(saveimr_name,real_A)

