"""
Module: utils.py
==================

This module contains the functions needed to facilitate the pre-processing and loading of the dataset

Author:
    Jorge Andrés Hernández Galeano

Date:
    2023-08-27
"""

# Pytorch and Torchvision Imports
import torch
import torch.nn as nn

import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms.functional import to_pil_image


from Resources.wips_dataset import wipsDataset

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'        

mean_data = [0.4952,0.3626,0.1201]
std_data = [0.3231,0.2541,0.1689]

#transforms.Normalize(mean=mean_data, std=std_data)
#transform = transforms.Compose(transforms.ToTensor())




"""
get_dataset
Required parameters: category - specimens category, if not provided, dataset with all categories is returned
Optional parameters: transform - transform to apply to images at loading the dataset
Output: dataset instance
"""
def get_dataset(images_root = images_root,images_reference = images_reference,category = -1, transform = None):
    dataset = wipsDataset(images_reference,images_root, category = category, transform = transform)
    return dataset




#function used by the dataloader, it applies resizing(116x256) and padding to dataset images
def custom_collate(batch):
    resized_batch = []
    categories = []
    for sample in batch:
        image = sample['image']
        pil_image = to_pil_image(image)  # Convert to PIL image
        #resized_image = transforms.Resize((232, 512))(pil_image)
        resized_image = transforms.Resize((116, 256))(pil_image)
        #resized_image = transforms.Resize((116, 256))(pil_image)
        #padded_image = transforms.Pad((0,0,0,280))(resized_image)
        #padded_image = transforms.Pad((0,0,0,24))(resized_image)
        padded_image = transforms.Pad((0,0,0,140))(resized_image)
        resized_tensor = transforms.ToTensor()(padded_image)  # Convert back to tensor
        resized_batch.append(resized_tensor)
        categories.append(torch.tensor(sample['category']))
        
    #stacked_batch_images = torch.tensor(resized_batch)
    #stacked_categories = torch.tensor(categories)

    stacked_batch_images = torch.stack(resized_batch)
    stacked_categories = torch.stack(categories)

    return stacked_batch_images, stacked_categories

"""
get_dataloader

Required parameters: category - specimens category, if not provided, dataset with all categories is returned
                     batch_size for training
                     drop_last - True: it drops last batch if size is less than batch_size
                     
Optional parameters: transform - transform to apply to images at loading the dataset
Output: dataloader for training
"""

def get_dataloader(images_root = images_root, images_reference = images_reference ,batch_size = 16,train__size_factor = 1,category = -1, drop_last = False, transform = None):
    #transform = transforms.RandomHorizontalFlip()
    dataset = get_dataset(images_root = images_root, images_reference = images_reference, category = category, transform = transform)
    train_size = int(train__size_factor * len(dataset))
    train_dataloader = data.DataLoader(dataset, batch_size, shuffle= True, collate_fn= custom_collate, drop_last = drop_last)
    return train_dataloader




class PackedWipsDataset(Dataset):
    def __init__(self, original_dataset, packing_num=2):
        self.original_dataset = original_dataset
        self.packing_num = packing_num
        self.length = (len(self.original_dataset)//packing_num)

    def __getitem__(self, index):
        combined_images = []
        for i in range(self.packing_num):
            pil_image = to_pil_image(self.original_dataset[index * self.packing_num + i]["image"])
            resized_image = transforms.Resize((116, 256))(pil_image)
            padded_image = transforms.Pad((0,0,0,140))(resized_image)
            resized_tensor = transforms.ToTensor()(padded_image)
            combined_images.append(resized_tensor)
        combined_image = torch.cat(combined_images, dim=0)
        return combined_image
    def __len__(self):
        return self.length



"""
get_packed_dataloader

Required parameters: category - specimens category, if not provided, dataset with all categories is returned
                     batch_size for training
                     drop_last - True: it drops last batch if size is less than batch_size
                     packing_num  - number of packed images in a batch sample
                     
Optional parameters: transform - transform to apply to images at loading the dataset
Output: dataloader of packed for PACGAN training
"""    
def get_packed_dataloader(images_root = images_root, images_reference = images_reference ,batch_size = 16,train__size_factor = 1,category = -1, drop_last = False, packing_num = 2):
    dataset = get_dataset(images_root = images_root, images_reference = images_reference, category = category)
    packed_dataset = PackedWipsDataset(dataset, packing_num=packing_num)
    dataloader = data.DataLoader(packed_dataset, batch_size, shuffle= True, drop_last = drop_last)
    return dataloader
