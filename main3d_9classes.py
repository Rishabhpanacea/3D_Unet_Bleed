import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Custom modules and configuration
from src.configuration.config import (
    TrainingDir, batch_size, num_epochs, num_workers,
    pin_memory, LEARNING_RATE
)
from src.Dataset.dataset import CustomDatasetHW_3D
from src.utils.losses import GeneralizedDiceLoss
from src.utils.utils import custom_collate_Variable_HW
from src.Models.D_UNet import UNet3D

# Combined loss: Dice + BCE, with targets cast to float for BCE.
def combined_loss(outputs, targets, dice_weight=0.5):
    dice_loss = GeneralizedDiceLoss()(outputs, targets)
    # Cast targets to float to match the output dtype.
    bce_loss = F.binary_cross_entropy_with_logits(outputs, targets.float())
    return dice_weight * dice_loss + (1 - dice_weight) * bce_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, scheduler, and AMP scaler using updated torch.amp syntax.
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler(device=device.type)

    # Prepare dataset and DataLoader
    images_dir = os.path.join(TrainingDir, 'Images')
    masks_dir = os.path.join(TrainingDir, 'Masks')
    dataset = CustomDatasetHW_3D(images_dir, masks_dir)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        collate_fn=custom_collate_Variable_HW,
    )

    best_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Use torch.amp.autocast with updated syntax.
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = combined_loss(outputs, targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        scheduler.step()

        # Dummy validation; replace with your actual validation evaluation.
        val_loss = avg_loss

        if val_loss < best_loss:
            best_loss = val_loss
            best_checkpoint_path = os.path.join(TrainingDir, f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Best checkpoint saved: {best_checkpoint_path}")

        checkpoint_path = os.path.join(TrainingDir, f"model_checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
