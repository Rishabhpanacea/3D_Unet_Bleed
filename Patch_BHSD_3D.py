from src.Dataset.dataset import Single_Volume_patch_Class_3D
from src.configuration.config import (
    datadict, TrainingDir, batch_size, num_epochs, num_workers,
    pin_memory, LEARNING_RATE, IMAGE_HEIGHT, IMAGE_WIDTH
)
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from src.Models.D_UNet import UNet3D
import os









def main():
    # Define data augmentation and transformation
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

    # Initialize model and loss function
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    lossfn = nn.BCEWithLogitsLoss()

    # Prepare dataset and DataLoader
    mask_dir = os.path.join(TrainingDir, 'ground truths/ID_0b10cbee_ID_f91d6a7cd2.nii.gz')
    image_dir = os.path.join(TrainingDir, 'images/ID_0b10cbee_ID_f91d6a7cd2.nii.gz')
    dataset = Single_Volume_patch_Class_3D(image_dir, mask_dir, transform=train_transform)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Set up Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
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

    print("Training complete!")

if __name__ == "__main__":
    main()
