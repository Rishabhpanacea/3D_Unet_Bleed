import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.configuration.config import datadict

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

        return ImageVolume, Maskvolume
