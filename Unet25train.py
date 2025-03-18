import os
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from src.Models.D_UNet import UNet2D, ResidualUNet2D
from src.configuration.config import datadict, IMAGE_HEIGHT, IMAGE_WIDTH, TrainingDir, batch_size, num_epochs, num_workers, pin_memory, LEARNING_RATE
from src.Dataset.dataset import CustomDataset2D
from src.utils.utils import custom_collate_2D, save_prediction, check_accuracy
import torch.optim as optim
import torch.nn as nn



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    image_dir = os.path.join(TrainingDir, 'Images')
    mask_dir = os.path.join(TrainingDir, 'Masks')
    data = CustomDataset2D(image_dir, mask_dir,transform = train_transform)

    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        collate_fn=custom_collate_2D
    )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_channels=3, out_channels=9, f_maps=128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")


    # lossfn = DiceLoss()
    lossfn = nn.BCEWithLogitsLoss()





    # Training loop
    check_accuracy(train_loader, model, device="cuda")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()


            outputs = model(inputs)
            loss = lossfn(outputs, targets)
        
            # Backpropagation with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        checkpoint_path = f"model_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        if epoch%10 == 0:
            check_accuracy(train_loader, model, device="cuda")

    print("Training complete!")
if __name__ == "__main__":
    main()
