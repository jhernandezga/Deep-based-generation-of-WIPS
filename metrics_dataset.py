# General Imports
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
#

from Resources.wips_dataset_v01 import wipsDataset

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])
dataset = wipsDataset(images_reference,images_root,transform = transform)

def custom_collate(batch):
    resized_batch = []
    for sample in batch:
        image = sample['image']
        image.type(torch.float32)
        pil_image = to_pil_image(image)  # Convert to PIL image
        resized_image = transforms.Resize((116, 256))(pil_image)
        resized_tensor = transforms.ToTensor()(resized_image)  # Convert back to tensor
        resized_batch.append(resized_tensor)
    stacked_batch = torch.stack(resized_batch)

    return {'image': stacked_batch}

batch_size = 32
dataloader = data.DataLoader(dataset, batch_size, collate_fn=custom_collate)
mean = 0.
std = 0.

#for sample in dataloader:
 #   images = sample['image']

"""for i in range(5):
    sample = dataset[i]
    image = sample['image']
    print(image.shape)

    image = sample['image']
    print(image.shape)"""
count = 0    
for sample in dataloader:
    try:
        batch_samples = sample['image'].size(0)
        images = sample['image']
        
        images.to(device)
        #if not (torch.equal(torch.Tensor([1,3,116,256]), torch.Tensor(list(images.shape)))):
        #    print(count)
         #   break
        images = sample['image'].view(batch_samples, sample['image'].size(1), -1)
        mean += ((images.mean(2))).sum(0)
        std += ((images.std(2))).sum(0)
        count += 1
    except: 
        print(count)
    

mean /= len(dataloader.dataset)
std /= len(dataloader.dataset)
print("Mean: ")
print(mean)
print("Std:")
print(std)
print("--------------------end------------------------")

#print(dataset[1647]['image'].shape)
#plt.imshow(dataset[1647]['image'])
#plt.show(block=True)           
"""for sample in dataloader:
 batch_samples = sample['image'].size(0) # batch size (the last batch can have smaller size!)
    images = sample['image'].view(batch_samples, sample['image'].size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
"""


