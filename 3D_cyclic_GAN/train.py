#########################################################################################
# Training script for 3D Cyclic GAN l21
#
# The code framework is built on DeepSynth provided by 
# Prof. Edward J. Delp and his laboratory at Purdue University
#
# In our paper, we majorly revise the model to accomendate 3D inputs, 
# change the loss functions, and modify the model parameters. 
#
# J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
# "3D GAN image aynthesis and dataset quality assessment for bacterial biofilm", Bioinformatics 2022
# https://doi.org/10.1093/bioinformatics/btac529
#
# Jie Wang, VIVA lab
# Last update: Apr. 17, 2022
#########################################################################################

import time
from options.train_options import TrainOptions
from data_loader3d import Unaligned3d
from models.models import create_model
import torch.utils.data
import numpy as np
import os

lstarttime = time.time()
opt = TrainOptions().parse()
data_loader = Unaligned3d(opt)
dataset= torch.utils.data.DataLoader(
                    data_loader,
                    batch_size=opt.batchSize,
                    shuffle=not opt.serial_batches,
                    num_workers=int(opt.nThreads))
# to check the size of the image
#
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
#print(data_loader.__len__)
    

model = create_model(opt)
total_steps = 0

total_loss = ['epoch','total loss']

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            
            #print(errors['total'])
            total_loss = np.append(total_loss, [epoch, errors['total']],axis=0)
            t = (time.time() - iter_start_time) / opt.batchSize

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

lendtime = time.time()
print('This is the end of learning, Time Taken: %d sec', lendtime - lstarttime)

lossforplot = os.path.join(opt.checkpoints_dir, opt.name, 'totalloss_log.csv')
np.savetxt(lossforplot, total_loss, fmt='%s')