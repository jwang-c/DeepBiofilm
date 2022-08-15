#########################################################################################
# Test model for 3D Cyclic GAN l21
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


from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import numpy as np

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G_A', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_numpy_images(self):
        real_A = self.input_A[0].cpu().float().detach().numpy()
        fake_B = self.fake_B[0].cpu().float().detach().numpy()

        [C_r,D_r,H_r,W_r] = real_A.shape
        A_temp = real_A.reshape(D_r,H_r,W_r)
        [C_f,D_f,H_f,W_f] = fake_B.shape
        B_temp = fake_B.reshape(D_f,H_f,W_f)

        #print(np.amin(B_temp))
        #print(np.amax(B_temp))
        A = (A_temp-np.amin(A_temp))/(np.amax(A_temp)-np.amin(A_temp)) * 255.0
        B = (B_temp-np.amin(B_temp))/(np.amax(B_temp)-np.amin(B_temp)) * 255.0
        rA = A.astype(np.uint8) # reconstructed A
        fB = B.astype(np.uint8) # reconstructed B
        #print(rA.shape)
        #print(rA.dtype)
        #print(fB.shape)
        #print(fB.dtype)
        return rA,fB
        

