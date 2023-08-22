import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import *
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor() 
])

category = 56

dataloader = get_dataloader(transform=transform, category=category, batch_size=1)

channel_sums = torch.zeros((1,3))  # Assuming RGB images
channel_std_sums = torch.zeros((1,3))

for batch in dataloader:
    image,_ = batch
    mean, std = image.mean([2,3]), image.std([2,3])
    channel_sums += mean
    channel_std_sums += std

num_samples = len(dataloader)
mean = channel_sums / num_samples
std = channel_std_sums / num_samples

print("Mean:", mean)
print("Standard Deviation:", std)

transform = transforms.Compose([
    transforms.Resize(size=(256,116)),
    transforms.ToTensor()     
])

dataloader2 = get_dataloader(batch_size=1,transform=transform,category=category)

imagex = next(iter(dataloader2))[0][0]
mean = mean.tolist()[0]
std = std.tolist()[0]


t = transforms.Compose([
    transforms.Normalize(mean=mean, std = std),
])

imagex = t(imagex)
t2 = transforms.ToPILImage()
imagex = t2(imagex)
imagex = np.asarray(imagex)

plt.imshow(imagex)
