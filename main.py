import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from src.configuration.config import datadict, TrainingDir, batch_size, num_epochs,num_workers,pin_memory, LEARNING_RATE
from src.Dataset.dataset import CustomDatasetHW, CustomDatasetHWD, CustomDataset, CustomDatasetHW_new, CustomDatasetHW_validation
from src.utils.losses import BCEDiceLoss, DiceLoss, GeneralizedDiceLoss, WeightedCrossEntropyLoss, WeightedSmoothL1Loss
from src.configuration.config import IMAGE_HEIGHT, IMAGE_WIDTH
from src.utils.utils import custom_collate, custom_collate_Variable_HW
from src.Models.D_UNet import UNet3D
import torch.optim as optim

if __name__ == "__main__":
    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DunetModel = UNet3D(in_channels=1, out_channels=1).to(device)
    lossfun = GeneralizedDiceLoss()


    ImagesDir = os.path.join(TrainingDir, 'Images')
    MasksDir = os.path.join(TrainingDir, 'Masks')
    # print(os.listdir(TrainingDir))
    data = CustomDatasetHW(ImagesDir, MasksDir, transform=train_transform)


    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        collate_fn=custom_collate_Variable_HW,
    )






    optimizer = optim.Adam(DunetModel.parameters(), lr=LEARNING_RATE)


    for epoch in range(num_epochs):
        DunetModel.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = DunetModel(inputs)
            loss = lossfun(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")
        
        # Save model checkpoint
        torch.save(DunetModel.state_dict(), f"model_checkpoint.pth")
        print(f"Checkpoint saved for epoch {epoch+1}")

    print("Training complete!")