import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from src.configuration.config import datadict, newDatadict
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

        self.output_size = output_size  # (H, W)
        self.output_depth = (IMAGE_HEIGHT, IMAGE_WIDTH)   # Depth for resizing

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







class CustomDatasetHW_3D_9classes(Dataset):
    def __init__(self, image_dir, mask_dir, datadict=datadict, output_size=(256, 256), output_depth=5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict if datadict else {}
        self.reversed_dict = {v: k for k, v in self.datadict.items()}

        self.output_size = output_size  # (H, W)
        self.output_depth = (IMAGE_HEIGHT, IMAGE_WIDTH)   # Depth for resizing

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
        # newMaskVolume = np.stack([np.argmax(Maskvolume[:, i, :, :], axis=0) for i in range(Maskvolume.shape[1])], axis=0)
        # newMaskVolume = np.expand_dims(newMaskVolume, axis=0)
        newMaskVolume = Maskvolume

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

        return torch.tensor(resized_images).unsqueeze(0), torch.tensor(resized_masks)
    








class CustomDataset2D(Dataset):
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
        count = 0
        for i in range(len(self.series)):
            first_folder = os.listdir(os.path.join(self.mask_dir, self.series[i]))[0]
            folder_path = os.path.join(self.mask_dir, self.series[i], first_folder)
            series_length = len(os.listdir(folder_path))
            count = count + series_length
        return count


    def transform_volume(self, image_volume, mask_volume):
        # print(image_volume.transpose(1, 2, 0).shape)
        # print(mask_volume.transpose(1, 2, 0).shape)
        transformed = self.transform(
                image=image_volume.transpose(1, 2, 0), 
                mask=mask_volume.transpose(1, 2, 0)  # Change (9, 512, 512) -> (512, 512, 9)
            )
        images = transformed['image']
        masks = transformed['mask'].permute(2, 0, 1)

        # print(images.shape)
        # print(masks.shape)

        return images , masks
        # transformed_images = []
        # transformed_masks = []
        
        # for i in range(image_volume.shape[0]):  # Iterate over 3 slices
        #     # print(mask_volume[:, i].shape, "  ", image_volume[i].shape)  # Debugging output
            
        #     transformed = self.transform(
        #         image=image_volume[i], 
        #         mask=mask_volume[:, i].transpose(1, 2, 0)  # Change (9, 512, 512) -> (512, 512, 9)
        #     )
            
        #     transformed_images.append(transformed['image'])  # Shape: (512, 512)
        #     transformed_masks.append(transformed['mask'].permute(2, 0, 1))  # Convert (512, 512, 9) -> (9, 512, 512)
        
        # transformed_image_volume = torch.stack(transformed_images)  # Shape: (3, 256, 256)
        # transformed_mask_volume = torch.stack(transformed_masks)  # Shape: (9, 3, 256, 256)
        
        # return transformed_image_volume, transformed_mask_volume

        
    def __getitem__(self, index):
        # print("log1")
        count = 0
        index = index + 1
        for i in range(len(self.series)):
            first_folder = os.listdir(os.path.join(self.mask_dir, self.series[i]))[0]
            folder_path = os.path.join(self.mask_dir, self.series[i], first_folder)
            series_length = len(os.listdir(folder_path))

            if count+series_length > index:
                self.series_index = i
                index = index-count-1
                break
            elif count+series_length == index:
                self.series_index = i
                index = series_length - 1
                break
            else:
                count = count + series_length

        # print("log2")
            
        Maskvolume = []
        ImageVolume = []
        flag = 0
        for key in range(len(self.reversed_dict.keys())):
            catag = self.reversed_dict[key]
            Maskcatgvolume = []
            Masks = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[self.series_index], catag)
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

        # print("log3")
        
        newMaskVolume = []
        for i in range(Maskvolume.shape[1]):
            newMaskVolume.append(np.argmax(Maskvolume[:,i,:,:] , axis=0))
        newMaskVolume = np.stack(newMaskVolume, axis=0)
        
        newMaskVolume[newMaskVolume>0] = -1
        newMaskVolume[newMaskVolume == 0] = 1
        newMaskVolume[newMaskVolume == -1] = 0
        
        for i in range(Maskvolume.shape[1]):
            Maskvolume[0,i,:,:] = Maskvolume[0,i,:,:] + newMaskVolume[i,:,:]


        # print("log4")



        newImageVolume = []
        newMaskVolume = []
        empty_slice = np.zeros(ImageVolume[0,:,:].shape)
        # empty_slice_mask = np.zeros(Maskvolume[:,0,:,:].shape)
        middleslice = ImageVolume[index,:,:]
        middlesliceMask = Maskvolume[:,index,:,:]

        # print("log5")
        
        if index == 0:
            if ImageVolume.shape[0] == 1:
                newImageVolume.append(empty_slice)
                newImageVolume.append(middleslice)
                newImageVolume.append(empty_slice)
                newImageVolume = np.stack(newImageVolume, axis=0)
                
                # newMaskVolume.append(empty_slice_mask)
                # newMaskVolume.append(middlesliceMask)
                # newMaskVolume.append(empty_slice_mask)
                # newMaskVolume = np.stack(newMaskVolume, axis=1)
            else:
                lastslice = ImageVolume[index+1,:,:]
                newImageVolume.append(empty_slice)
                newImageVolume.append(middleslice)
                newImageVolume.append(lastslice)
                newImageVolume = np.stack(newImageVolume, axis=0)

                
                # lastsliceMask = Maskvolume[:,index+1,:,:]
                # newImageVolume.append(empty_slice_mask)
                # newImageVolume.append(middlesliceMask)
                # newImageVolume.append(lastsliceMask)
                # newMaskVolume = np.stack(newMaskVolume, axis=1)
                
        elif index == (ImageVolume.shape[0]-1):
            firstslice = ImageVolume[index-1,:,:]
            newImageVolume.append(firstslice)
            newImageVolume.append(middleslice)
            newImageVolume.append(empty_slice)
            newImageVolume = np.stack(newImageVolume, axis=0)

            # firstsliceMask = Maskvolume[:,index-1,:,:]
            # newMaskVolume.append(firstsliceMask)
            # newMaskVolume.append(middlesliceMask)
            # newMaskVolume.append(empty_slice_mask)
            # newMaskVolume = np.stack(newMaskVolume, axis=1)

        else:
            firstslice = ImageVolume[index-1,:,:]
            lastslice = ImageVolume[index+1,:,:]
            newImageVolume.append(firstslice)
            newImageVolume.append(middleslice)
            newImageVolume.append(lastslice)
            newImageVolume = np.stack(newImageVolume, axis=0)


            # firstsliceMask = Maskvolume[:,index-1,:,:]
            # lastsliceMask = Maskvolume[:,index+1,:,:]
            # newMaskVolume.append(firstsliceMask)
            # newMaskVolume.append(middlesliceMask)
            # newMaskVolume.append(lastsliceMask)
            # newMaskVolume = np.stack(newMaskVolume, axis=1)

        # print("log6")

        
        if self.transform is not None:
            transformed_image_volume, transformed_mask_volume = self.transform_volume(newImageVolume, middlesliceMask)
            

        # return image, mask
        return transformed_image_volume, transformed_mask_volume
            
        return newImageVolume, middlesliceMask
        # return ImageVolume ,Maskvolume






class BHSD_3D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, datadict=newDatadict):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict
        reversed_dict = {v: k for k, v in datadict.items()}
        self.reversed_dict = reversed_dict

    def transform_volume(self, image_volume, mask_volume):
        transformed = self.transform(
                image=image_volume, 
                mask=mask_volume
            )
        images = transformed['image']
        masks = transformed['mask'].permute(2, 0, 1)
        return images , masks

    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        nii_segementation = nib.load(os.path.join(self.mask_dir, self.images[index]))
        nii_image = nib.load(os.path.join(self.image_dir, self.images[index]))
        
        # Get the image data as a NumPy array
        image_data = nii_image.get_fdata()
        segementation_data = nii_segementation.get_fdata()

        if self.transform is not None:
            transformed_image_volume, transformed_mask_volume = self.transform_volume(image_data, segementation_data)

        transformed_image_volume = transformed_image_volume.unsqueeze(0)
        transformed_mask_volume = transformed_mask_volume.unsqueeze(0)
        return transformed_image_volume, transformed_mask_volume