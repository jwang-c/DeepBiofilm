#########################################################################################
# 3D cell segmentation using V-net/3D u-net in the following paper:
#
# example for training with GAN-envolved dataset
#
# J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
# "3D GAN image aynthesis and dataset quality assessment for bacterial biofilm", 2022
#
# Jie Wang, VIVA lab
# Last update: Apr. 17, 2022
#
# We referred to the u-net training code framework from: 
# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet
#########################################################################################

import torch
#from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
from Unet3d import UNet
from utils_real import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    Dice_loss
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 200
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'Dataset/simulation_GANl21/Train_tif/images/'
TRAIN_MASK_DIR = 'Dataset/simulation_GANl21/Train_tif/labels/'
#VAL_IMG_DIR = 'Dataset/simulation_GANl21/val_tif/images/'
#VAL_MASK_DIR = 'Dataset/simulation_GANl21/val_tif/labels/'

VAL_IMG_DIR = 'Dataset/val_real_gt/images/'
VAL_MASK_DIR = 'Dataset/val_real_gt/labels/'

def train_fn(loader, model, optimizer, loss_fn):
    #loop = tqdm(loader)
    loop = loader
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update tqdm loop
        #loop.set_postfix(loss=loss.item())


def main():

    train_transform = None
    val_transforms = None
    model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    #loss_fn = 1
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint_50l21_300.pth.tar"), model)

    total_accuracies = [111,222,333] # just for initialization
    [acc,dice] = check_accuracy(val_loader, model, device=DEVICE)
    total_accuracies = np.append(total_accuracies, [-1, acc, dice],axis=0) 

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch = {epoch}")
        train_fn(train_loader, model, optimizer, loss_fn)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        [acc, dice] = check_accuracy(val_loader, model, device=DEVICE)
        total_accuracies = np.append(total_accuracies, [epoch, acc, dice],axis=0)
        
        # print some examples to a folder
    #save_predictions_as_imgs(
    #    val_loader, model, folder="saved_images_real_l21_gt/", device=DEVICE
    #)
    lossforplot = ('testacc_log_real_50l21_300.csv')
    np.savetxt(lossforplot, total_accuracies, fmt='%1.3f')

if __name__ == "__main__":
    main()