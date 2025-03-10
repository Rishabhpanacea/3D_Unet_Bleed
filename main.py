import torch
import os
from torch.utils.data import DataLoader
from src.Models.D_UNet import UNet3D
from src.utils.losses import BCEDiceLoss, DiceLoss, GeneralizedDiceLoss, WeightedCrossEntropyLoss, WeightedSmoothL1Loss
from src.Dataset.dataset import CustomDataset
from src.utils.utils import custom_collate
from src.configuration.config import TrainingDir, batch_size , num_workers ,pin_memory, LEARNING_RATE, num_epochs
from src.Training.Train import train_fn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

if __name__ == "__main__":

    train_transform = A.Compose(
    [
        ToTensorV2(),
    ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DunetModel = UNet3D(in_channels=1, out_channels=9).to(device)

    # Loss Function
    loss_function = BCEDiceLoss()

    # Optimizer
    optimizer = optim.Adam(DunetModel.parameters(), lr=LEARNING_RATE)

    ImagesDir = os.path.join(TrainingDir, 'Images')
    MasksDir = os.path.join(TrainingDir, 'Masks')
    print(os.listdir(TrainingDir))
    data = CustomDataset(ImagesDir, MasksDir ,transform= train_transform)

    batch_size = 2
    num_workers = 0
    pin_memory = True
    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        collate_fn=custom_collate
    )

    # for batch_features, batch_labels in train_loader:
    #     print(batch_features.shape)
    #     print(batch_labels.shape)
    #     print("-"*50)
    # print(data[0])
    for epoch in range(num_epochs):
        DunetModel.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = DunetModel(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")
        
        # Save model checkpoint
        torch.save(DunetModel.state_dict(), f"model_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved for epoch {epoch+1}")

    print("Training complete!")