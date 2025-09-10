import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from src.utils.losses import BCEDiceLoss
# Load configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "model": {
        "in_channels": 1,
        "out_channels": 1,
        "layer_order": "gcr",
        "f_maps": 32,
        "num_groups": 8,
        "final_sigmoid": True
    },
    "loss": {
        "name": "BCEDiceLoss",
        "skip_last_target": True
    },
    "optimizer": {
        "learning_rate": 0.0002,
        "weight_decay": 0.00001
    },
    "eval_metric": {
        "name": "BoundaryAdaptedRandError",
        "threshold": 0.4,
        "use_last_target": True,
        "use_first_input": True
    },
    "lr_scheduler": {
        "name": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 30
    },
    "trainer": {
        "max_num_epochs": 1000,
        "max_num_iterations": 150000,
        "validate_after_iters": 1000,
        "log_after_iters": 500,
        "checkpoint_dir": "CHECKPOINT_DIR"
    },
    "loaders": {
        "num_workers": 32,
        "train": {
            "file_paths": ["PATH_TO_TRAIN_DIR"],
            "batch_size": 2
        },
        "val": {
            "file_paths": ["PATH_TO_VAL_DIR"],
            "batch_size": 2
        }
    }
}

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
# Initialize model
model = UNet3D(
    in_channels=config["model"]["in_channels"],
    out_channels=config["model"]["out_channels"],
    f_maps=config["model"]["f_maps"],
    num_groups=config["model"]["num_groups"],
    final_sigmoid=config["model"]["final_sigmoid"]
).cuda()

# Loss function
criterion = BCEDiceLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=config["optimizer"]["learning_rate"],
    weight_decay=config["optimizer"]["weight_decay"]
)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=30)

mask_dir = os.path.join(TrainingDir, 'ground truths/ID_0b10cbee_ID_f91d6a7cd2.nii.gz')
image_dir = os.path.join(TrainingDir, 'images/ID_0b10cbee_ID_f91d6a7cd2.nii.gz')

# Load datasets
train_dataset = Single_Volume_patch_Class_3D(image_dir, mask_dir, transform=train_transform)
val_dataset = Single_Volume_patch_Class_3D(image_dir, mask_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8)

# Training loop
for epoch in range(config["trainer"]["max_num_epochs"]):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # inputs, targets = batch["raw"].cuda(), batch["label"].cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(loss.item())
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # inputs, targets = batch["raw"].cuda(), batch["label"].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
    
    scheduler.step(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss}")
    
    # Save model checkpoint
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{config['trainer']['checkpoint_dir']}/checkpoint_epoch_{epoch}.pth")
