"""
Module wips_dataset.py
==================

It defines the custom dataset loading structure of WIPs dataset for Pytorch

Author:
    Jorge Andrés Herández Galeano
    https://github.com/jhernandezga
    jhernandezga@unal.edu.co

Date:
    2023-08-26
"""


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torchvision.transforms.functional import to_pil_image



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class wipsDataset(Dataset):
    
    #Category default: -1 -> whole dataset
    def __init__(self, csv_file, root_dir, transform=None, category = -1):
        
        if category!= -1:
            self.image_frame = pd.read_csv(csv_file)
            if category not in self.image_frame['category'].values:
                raise ValueError('Category not found: Category [1-144]')
            
            self.image_frame.sort_values(by=['category','photo'], inplace= True)
            self.image_frame = self.image_frame[self.image_frame['category'] == category]
            
            if len(self.image_frame) == 0:
                raise ValueError('Category not found: Category [1-144]')
        else:
            self.image_frame = pd.read_csv(csv_file,header=None,skiprows= 1)
            
        
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        
        img_name = self.image_frame.iloc[idx, 0]                     
        img_name = os.path.join(self.root_dir, ".".join([str(img_name),'jpg']))
        
        category = self.image_frame.iloc[idx,2]
        sex = self.image_frame.iloc[idx,3]
        image = io.imread(img_name)
        
        # Convert image to uint8
        image = image.astype(np.uint8)
        
        # Check if the image has an alpha channel (RGBA)
        if image.shape[-1] == 4:
            # Convert RGBA image to RGB
            image = color.rgba2rgb(image)
        
        sample = {'image': image, 'img_name':img_name, 'category': category, 'sex':sex}
        
        if self.transform:
            img = to_pil_image(sample['image'].astype(np.uint8))
            sample['image'] = self.transform(img)
            
        return sample


