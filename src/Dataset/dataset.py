import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from src.configuration.config import datadict
from src.configuration.config import IMAGE_HEIGHT, IMAGE_WIDTH

import torchio as tio

class CustomDatasetHW_3D(Dataset):
    def __init__(self, image_dir, mask_dir, datadict=datadict, output_size=(256, 256), output_depth=5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict if datadict else {}
        self.reversed_dict = {v: k for k, v in self.datadict.items()}

        self.output_size = (IMAGE_HEIGHT, IMAGE_WIDTH) # (H, W)
        self.output_depth = output_depth  # Depth for resizing

        # Define 3D Transformations using TorchIO
        self.tio_transform = tio.Compose([
            tio.RandomAffine(scales=(0.9, 1.1), degrees=35),  # Random rotation & scaling
            tio.RandomFlip(axes=(0, 1), p=0.5),  # Flip along horizontal & vertical axes
            tio.ZNormalization(),  # Normalize
        ])

    def __len__(self):
        return len(self.series)

    def resize_volume(self, volume, new_depth):
        """Resize depth using linear interpolation."""
        d, h, w = volume.shape
        resized_volume = np.zeros((new_depth, h, w), dtype=volume.dtype)

        for i in range(new_depth):
            orig_idx = int(i * (d / new_depth))  # Interpolation
            resized_volume[i] = volume[orig_idx]

        return resized_volume

    def __getitem__(self, index):
        Maskvolume = []
        ImageVolume = []
        flag = 0

        for key in range(len(self.reversed_dict.keys())):
            category = self.reversed_dict[key]
            Maskcatgvolume = []
            Masks = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index], category)
            MasksList = sorted(os.listdir(Masks))
            
            for msk in MasksList:
                pngMask = np.array(Image.open(os.path.join(Masks, msk)))
                Maskcatgvolume.append(pngMask)
    
                if msk in self.images and flag == 0:
                    pngimage = np.array(Image.open(os.path.join(self.image_dir, msk)))
                    ImageVolume.append(pngimage)
            flag = 1

            Maskcatgvolume = np.stack(Maskcatgvolume, axis=0)
            Maskvolume.append(Maskcatgvolume)

        Maskvolume = np.stack(Maskvolume, axis=0)
        ImageVolume = np.stack(ImageVolume, axis=0)
        ImageVolume = np.expand_dims(ImageVolume, axis=0)

        # Convert multi-category masks to a single-class mask
        newMaskVolume = np.stack([np.argmax(Maskvolume[:, i, :, :], axis=0) for i in range(Maskvolume.shape[1])], axis=0)
        newMaskVolume = np.expand_dims(newMaskVolume, axis=0)

        # Apply 3D Transformations using TorchIO
        image_tensor = torch.tensor(ImageVolume, dtype=torch.float32)
        mask_tensor = torch.tensor(newMaskVolume, dtype=torch.int64)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor),
            mask=tio.LabelMap(tensor=mask_tensor),
        )

        transformed_subject = self.tio_transform(subject)
        transformed_image = transformed_subject["image"].tensor.numpy()
        transformed_mask = transformed_subject["mask"].tensor.numpy()

        # Resize to the desired shape
        resized_images = np.array([cv2.resize(img, self.output_size, interpolation=cv2.INTER_LINEAR) for img in transformed_image[0]])
        resized_masks = np.array([cv2.resize(mask, self.output_size, interpolation=cv2.INTER_NEAREST) for mask in transformed_mask[0]])

        return torch.tensor(resized_images).unsqueeze(0), torch.tensor(resized_masks).unsqueeze(0)


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, datadict=datadict):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict
        reversed_dict = {v: k for k, v in datadict.items()}
        self.reversed_dict = reversed_dict
    def __len__(self):
        return len(self.series)

    def __getitem__(self, index):
        Maskvolume = []
        ImageVolume = []
        print(self.series[index])
        flag = 0
        for key in range(len(self.reversed_dict.keys())):
            catag = self.reversed_dict[key]
            Maskcatgvolume = []
            Masks = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index], catag)
            MasksList = os.listdir(Masks)
            MasksList = sorted(MasksList)
            
            for msk in MasksList:
                pngMask = Image.open(os.path.join(Masks, msk))
                pngMask = np.array(pngMask)
                Maskcatgvolume.append(pngMask)
    
                if msk in self.images and flag == 0:
                    pngimage = Image.open(os.path.join(self.image_dir ,msk))
                    pngimage = np.array(pngimage)
                    ImageVolume.append(pngimage)
            flag = 1
                    
            Maskcatgvolume = np.stack(Maskcatgvolume, axis = 0)
            Maskvolume.append(Maskcatgvolume)
            
        Maskvolume = np.stack(Maskvolume, axis = 0)
        ImageVolume = np.stack(ImageVolume, axis = 0)
        ImageVolume = np.expand_dims(ImageVolume, axis=0)

        ImageVolume = torch.tensor(ImageVolume, dtype=torch.float16)
        Maskvolume = torch.tensor(Maskvolume, dtype=torch.float16)



        return ImageVolume, Maskvolume
    



class CustomDatasetHWD(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, datadict=datadict,  output_size=(256, 256), output_depth=5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict
        reversed_dict = {v: k for k, v in datadict.items()}
        self.reversed_dict = reversed_dict

        self.output_size = (IMAGE_HEIGHT, IMAGE_WIDTH)  # (H, W)
        self.output_depth = output_depth  # New Depth

    def __len__(self):
        return len(self.series)


    def resize_volume(self, volume, new_depth):
        """Resize depth using linear interpolation."""
        d, h, w = volume.shape
        resized_volume = np.zeros((new_depth, h, w), dtype=volume.dtype)

        for i in range(new_depth):
            orig_idx = int(i * (d / new_depth))  # Interpolation
            resized_volume[i] = volume[orig_idx]

        return resized_volume

    def __getitem__(self, index):
        Maskvolume = []
        ImageVolume = []
        print(self.series[index])
        flag = 0
        for key in range(len(self.reversed_dict.keys())):
            catag = self.reversed_dict[key]
            Maskcatgvolume = []
            Masks = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index], catag)
            MasksList = os.listdir(Masks)
            MasksList = sorted(MasksList)
            
            for msk in MasksList:
                pngMask = Image.open(os.path.join(Masks, msk))
                pngMask = np.array(pngMask)
                Maskcatgvolume.append(pngMask)
    
                if msk in self.images and flag == 0:
                    pngimage = Image.open(os.path.join(self.image_dir ,msk))
                    pngimage = np.array(pngimage)
                    ImageVolume.append(pngimage)
            flag = 1
                    
            Maskcatgvolume = np.stack(Maskcatgvolume, axis = 0)
            Maskvolume.append(Maskcatgvolume)
            
        Maskvolume = np.stack(Maskvolume, axis = 0)
        ImageVolume = np.stack(ImageVolume, axis = 0)
        ImageVolume = np.expand_dims(ImageVolume, axis=0)
        newMaskVolume = []
        for i in range(Maskvolume.shape[1]):
            newMaskVolume.append(np.argmax(Maskvolume[:,i,:,:] , axis=0))
        newMaskVolume = np.stack(newMaskVolume, axis=0)
        newMaskVolume = np.expand_dims(newMaskVolume, axis=0)



        

        resized_images = np.array([cv2.resize(img, self.output_size, interpolation=cv2.INTER_LINEAR) for img in ImageVolume[0]])
        resized_masks = np.array([cv2.resize(mask, self.output_size, interpolation=cv2.INTER_NEAREST) for mask in newMaskVolume[0]])

        # Resize Depth
        resized_images = self.resize_volume(resized_images, self.output_depth)  # (New D, H, W)
        resized_masks = self.resize_volume(resized_masks, self.output_depth)  # (New D, H, W)

        return torch.tensor(resized_images).unsqueeze(0), torch.tensor(resized_masks).unsqueeze(0)





class CustomDatasetHW(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, datadict=datadict,  output_size=(256, 256), output_depth=5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict
        reversed_dict = {v: k for k, v in datadict.items()}
        self.reversed_dict = reversed_dict

        self.output_size = (IMAGE_HEIGHT, IMAGE_WIDTH)  # (H, W)
        self.output_depth = output_depth  # New Depth

    def __len__(self):
        return len(self.series)


    def resize_volume(self, volume, new_depth):
        """Resize depth using linear interpolation."""
        d, h, w = volume.shape
        resized_volume = np.zeros((new_depth, h, w), dtype=volume.dtype)

        for i in range(new_depth):
            orig_idx = int(i * (d / new_depth))  # Interpolation
            resized_volume[i] = volume[orig_idx]

        return resized_volume

    def __getitem__(self, index):
        Maskvolume = []
        ImageVolume = []
        # print(self.series[index])
        flag = 0
        for key in range(len(self.reversed_dict.keys())):
            catag = self.reversed_dict[key]
            Maskcatgvolume = []
            Masks = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index], catag)
            MasksList = os.listdir(Masks)
            MasksList = sorted(MasksList)
            
            for msk in MasksList:
                pngMask = Image.open(os.path.join(Masks, msk))
                pngMask = np.array(pngMask)
                Maskcatgvolume.append(pngMask)
    
                if msk in self.images and flag == 0:
                    pngimage = Image.open(os.path.join(self.image_dir ,msk))
                    pngimage = np.array(pngimage)
                    ImageVolume.append(pngimage)
            flag = 1
                    
            Maskcatgvolume = np.stack(Maskcatgvolume, axis = 0)
            Maskvolume.append(Maskcatgvolume)
            
        Maskvolume = np.stack(Maskvolume, axis = 0)
        ImageVolume = np.stack(ImageVolume, axis = 0)
        ImageVolume = np.expand_dims(ImageVolume, axis=0)
        newMaskVolume = []
        for i in range(Maskvolume.shape[1]):
            newMaskVolume.append(np.argmax(Maskvolume[:,i,:,:] , axis=0))
        newMaskVolume = np.stack(newMaskVolume, axis=0)
        newMaskVolume = np.expand_dims(newMaskVolume, axis=0)



        

        resized_images = np.array([cv2.resize(img, self.output_size, interpolation=cv2.INTER_LINEAR) for img in ImageVolume[0]])
        resized_masks = np.array([cv2.resize(mask, self.output_size, interpolation=cv2.INTER_NEAREST) for mask in newMaskVolume[0]])

        # print('resized_images:-',resized_images.shape)
        # print('resized_masks:-',resized_masks.shape)
        # print(np.unique(resized_images))


        new_images = []
        new_masks = []
        if self.transform is not None:
            for slic in range(resized_images.shape[0]):
                image = resized_images[slic,:,:]
                mask = resized_masks[slic,:,:]

                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"].squeeze(0)
                mask = augmentations["mask"].squeeze(0)
                new_images.append(image)
                new_masks.append(mask)
        
        return torch.stack(new_images).unsqueeze(0), torch.stack(new_masks).unsqueeze(0)



        # return torch.tensor(resized_images).unsqueeze(0), torch.tensor(resized_masks).unsqueeze(0)










class CustomDatasetHW_new(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, datadict=None, output_size=(256, 256), output_depth=5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict if datadict is not None else {}
        self.reversed_dict = {v: k for k, v in self.datadict.items()}

        self.output_size = output_size  # (H, W)
        self.output_depth = output_depth  # New Depth

    def __len__(self):
        return len(self.series)

    def resize_volume(self, volume, new_depth):
        """Resize depth using linear interpolation."""
        d, h, w = volume.shape
        resized_volume = np.zeros((new_depth, h, w), dtype=volume.dtype)

        for i in range(new_depth):
            orig_idx = int(i * (d / new_depth))  # Interpolation
            resized_volume[i] = volume[orig_idx]

        return resized_volume

    def __getitem__(self, index):
        Maskvolume = []
        ImageVolume = []
        flag = 0

        for key in range(len(self.reversed_dict.keys())):
            catag = self.reversed_dict[key]
            Maskcatgvolume = []
            Masks = os.path.join(self.mask_dir, self.series[index], catag)
            MasksList = sorted(os.listdir(Masks))

            for msk in MasksList:
                pngMask = Image.open(os.path.join(Masks, msk)).convert("L")
                pngMask = np.array(pngMask)
                Maskcatgvolume.append(pngMask)

                if msk in self.images and flag == 0:
                    pngimage = Image.open(os.path.join(self.image_dir, msk)).convert("L")
                    pngimage = np.array(pngimage)
                    ImageVolume.append(pngimage)
            flag = 1

            Maskcatgvolume = np.stack(Maskcatgvolume, axis=0)
            Maskvolume.append(Maskcatgvolume)

        Maskvolume = np.stack(Maskvolume, axis=0)
        ImageVolume = np.stack(ImageVolume, axis=0)
        ImageVolume = np.expand_dims(ImageVolume, axis=0)

        # Convert multi-channel mask into a single-channel mask
        newMaskVolume = np.array([np.argmax(Maskvolume[:, i, :, :], axis=0) for i in range(Maskvolume.shape[1])])
        newMaskVolume = np.expand_dims(newMaskVolume, axis=0)

        # Resize
        resized_images = np.array([cv2.resize(img, self.output_size, interpolation=cv2.INTER_LINEAR) for img in ImageVolume[0]])
        resized_masks = np.array([cv2.resize(mask, self.output_size, interpolation=cv2.INTER_NEAREST) for mask in newMaskVolume[0]])

        # Apply transformation slice-wise
        transformed_images = []
        transformed_masks = []

        for img, mask in zip(resized_images, resized_masks):
            transformed = self.transform(image=img[..., np.newaxis], mask=mask[..., np.newaxis])
            transformed_images.append(transformed["image"].squeeze(0))  # Remove channel dimension after ToTensorV2
            transformed_masks.append(transformed["mask"].squeeze(0))

        # Convert lists to tensors
        transformed_images = torch.stack(transformed_images).unsqueeze(0)  # Add batch dimension
        transformed_masks = torch.stack(transformed_masks).unsqueeze(0)

        return transformed_images, transformed_masks












class CustomDatasetHW_validation(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, datadict=datadict,  output_size=(256, 256), output_depth=5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict
        reversed_dict = {v: k for k, v in datadict.items()}
        self.reversed_dict = reversed_dict

        self.output_size = (IMAGE_HEIGHT, IMAGE_WIDTH)  # (H, W)
        self.output_depth = output_depth  # New Depth

    def __len__(self):
        return len(self.series)


    def resize_volume(self, volume, new_depth):
        """Resize depth using linear interpolation."""
        d, h, w = volume.shape
        resized_volume = np.zeros((new_depth, h, w), dtype=volume.dtype)

        for i in range(new_depth):
            orig_idx = int(i * (d / new_depth))  # Interpolation
            resized_volume[i] = volume[orig_idx]

        return resized_volume

    def __getitem__(self, index):
        Maskvolume = []
        ImageVolume = []
        print(self.series[index])
        flag = 0
        for key in range(len(self.reversed_dict.keys())):
            catag = self.reversed_dict[key]
            Maskcatgvolume = []
            Masks = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index], catag)
            MasksList = os.listdir(Masks)
            MasksList = sorted(MasksList)
            
            for msk in MasksList:
                pngMask = Image.open(os.path.join(Masks, msk))
                pngMask = np.array(pngMask)
                Maskcatgvolume.append(pngMask)
    
                if msk in self.images and flag == 0:
                    pngimage = Image.open(os.path.join(self.image_dir ,msk))
                    pngimage = np.array(pngimage)
                    ImageVolume.append(pngimage)
            flag = 1
                    
            Maskcatgvolume = np.stack(Maskcatgvolume, axis = 0)
            Maskvolume.append(Maskcatgvolume)
            
        Maskvolume = np.stack(Maskvolume, axis = 0)
        ImageVolume = np.stack(ImageVolume, axis = 0)
        # ImageVolume = np.expand_dims(ImageVolume, axis=0)
        newMaskVolume = []
        for i in range(Maskvolume.shape[1]):
            newMaskVolume.append(np.argmax(Maskvolume[:,i,:,:] , axis=0))
        newMaskVolume = np.stack(newMaskVolume, axis=0)
        # newMaskVolume = np.expand_dims(newMaskVolume, axis=0)

        resized_images = ImageVolume
        resized_masks = newMaskVolume
        resized_images = resized_images.astype(np.float32)
        resized_masks = resized_masks.astype(np.uint8)

        # print('resized_images:-',resized_images.shape)
        # print('resized_masks:-',resized_masks.shape)

        # print(np.unique(resized_images))



        

        # resized_images = np.array([cv2.resize(img, self.output_size, interpolation=cv2.INTER_LINEAR) for img in ImageVolume[0]])
        # resized_masks = np.array([cv2.resize(mask, self.output_size, interpolation=cv2.INTER_NEAREST) for mask in newMaskVolume[0]])


        new_images = []
        new_masks = []
        if self.transform is not None:
            for slic in range(resized_images.shape[0]):
                image = resized_images[slic,:,:]
                mask = resized_masks[slic,:,:]

                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"].squeeze(0)
                mask = augmentations["mask"].squeeze(0)
                new_images.append(image)
                new_masks.append(mask)
        
        return torch.stack(new_images).unsqueeze(0), torch.stack(new_masks).unsqueeze(0)

