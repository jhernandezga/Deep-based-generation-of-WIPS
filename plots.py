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

root_generated = 'Generated_images/ResNet_wsdiv_51'
samples = 32
generated_samples_n =50

# Create a 4x4 grid of subplots
fig, axes = plt.subplots(8, 4, figsize=(12,11 ),tight_layout=True)

images = []
for i in range(samples):
    generated_sample = io.imread(os.path.join(root_generated, ".".join([str(random.randint(generated_samples_n)),'jpg'])))
    images.append(generated_sample)

for i, ax in enumerate(axes.flat):
    if i < len(images):
        image = images[i]
        ax.imshow(image)
        ax.axis('off')  # Turn off axis labels and ticks

# Adjust layout to prevent overlapping of subplots
plt.subplots_adjust(wspace=0.001, hspace=0.001)

# Show the grid of images
plt.show()






generated_sample = io.imread(os.path.join(root_generated, ".".join([str(i),'jpg'])))