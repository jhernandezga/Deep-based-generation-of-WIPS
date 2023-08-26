# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import os

import matplotlib.pyplot as plt
import numpy as np

import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from numpy import random

from Resources.wips_dataset import wipsDataset

from skimage import metrics, io
from skimage import transform as tf
from skimage.io import imsave, imread
from utils import *

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'

samples = 4
category = 15

generated_samples_n =50

dataset = get_dataset(images_root, images_reference,category=category)

n = len(dataset)


# Create a 4x4 grid of subplots
fig, axes = plt.subplots(1, 4, figsize=(7, 2),tight_layout=True)

images = []
for i in range(samples):
    sample = (tf.resize((dataset[random.randint(n)])['image'], (116,256), anti_aliasing= True, preserve_range=True)).astype(np.uint8)
    images.append(sample)

for i, ax in enumerate(axes.flat):
    if i < len(images):
        image = images[i]
        ax.imshow(image)
        ax.axis('off')  # Turn off axis labels and ticks

# Adjust layout to prevent overlapping of subplots
plt.subplots_adjust(wspace=0.001, hspace=0.001)

# Show the grid of images
plt.show()
