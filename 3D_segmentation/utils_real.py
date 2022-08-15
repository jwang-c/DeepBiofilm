#########################################################################################
# 3D cell segmentation using V-net/3D u-net in the following paper:
#
# J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
# "3D GAN image aynthesis and dataset quality assessment for bacterial biofilm", 2022
#
# Jie Wang, VIVA lab
# Last update: Apr. 17, 2022
#
# We referenced to the u-net code framework from: 
# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet
#########################################################################################

import torch
import torchvision
from dataset_real import BiofilmDataset
from torch.utils.data import DataLoader
from tifffile import imsave
import numpy as np

def save_checkpoint(state, filename="checkpoint/my_checkpoint_l21_300.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
    pin_memory,
):
    train_ds = BiofilmDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    val_ds = BiofilmDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            #preds = model(x)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    num_correct = float(num_correct)
    num_pixels = float(num_pixels)
    acc = num_correct / num_pixels * 100
    print(
        f"Got {num_correct}/{num_pixels} with acc {acc}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder, device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            #print(preds.shape)
            preds = (preds > 0.5).cpu().float().detach().numpy()
            [C_i,C_o,D,H,W] = preds.shape
            preds_img = preds.reshape(D,H,W)
            #print(preds_img.shape)
            #print(np.amin(preds_img))
            #print(np.amax(preds_img))
            preds_imgf = (preds_img-np.amin(preds_img))/(np.amax(preds_img)-np.amin(preds_img)) * 255.0
            preds_imgf = preds_imgf.astype(np.uint8)
            #preds_imgf = preds_img # for test_case, to keep 0/1
            imsave(f"{folder}/pred_{idx}.tif",preds_imgf)
        #torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

        #save the corresponding x and y
        xd = x.cpu().float().detach().numpy()
        [C_i,C_o,D,H,W] = xd.shape
        x_img = xd.reshape(D,H,W)   
        x_imgf = x_img * 255.0
        x_imgf = x_imgf.astype(np.uint8)
        imsave(f"{folder}/images_{idx}.tif",x_imgf) 

        yd = y.cpu().float().detach().numpy() 
        [C_i,C_o,D,H,W] = yd.shape
        y_img = yd.reshape(D,H,W)   
        y_imgf = y_img * 255.0
        y_imgf = y_imgf.astype(np.uint8)
        imsave(f"{folder}/labels_{idx}.tif",y_imgf) 

    model.train()
