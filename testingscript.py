import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from src.configuration.config import datadict, TrainingDir
from src.Dataset.dataset import CustomDatasetHW, CustomDatasetHWD, CustomDataset, CustomDatasetHW_new, CustomDatasetHW_validation,CustomDatasetHW_3D
from src.utils.losses import BCEDiceLoss, DiceLoss, GeneralizedDiceLoss, WeightedCrossEntropyLoss, WeightedSmoothL1Loss
from src.configuration.config import IMAGE_HEIGHT, IMAGE_WIDTH,batch_size, num_workers, pin_memory, LEARNING_RATE
from src.utils.utils import custom_collate, custom_collate_Variable_HW
from src.Models.D_UNet import UNet3D,ResidualUNetSE3D
import torch.optim as optim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DunetModel = UNet3D(in_channels=1, out_channels=1).to(device)
    DunetModel = ResidualUNetSE3D(in_channels=1, out_channels=1).to(device)
    # lossfun = GeneralizedDiceLoss()
    lossfun = GeneralizedDiceLoss()

    LEARNING_RATE = 0.0001

    optimizer = optim.Adam(DunetModel.parameters(), lr=LEARNING_RATE)

    # Set up Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler()

    # Define shape
    shape = (1, 1, 16, 64, 64)

    # Tensor with random values
    inputs = torch.rand(shape)

    # Probability of 1 occurring is very low (e.g., 0.05)
    probability_of_one = 0.0005  
    targets = torch.bernoulli(torch.full(shape, probability_of_one)).to(torch.int)

    print(torch.unique(targets, return_counts=True))
    DunetModel.train()
    for i in range(10000):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precisio/n
        with torch.cuda.amp.autocast():
            outputs = DunetModel(inputs)
            loss = lossfun(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if i%100 == 0:
            print(loss.item())


if __name__ == "__main__":
    main()
