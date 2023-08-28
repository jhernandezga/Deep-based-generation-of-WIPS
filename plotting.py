"""
Script:  plotting.py
==================

Script to plot images samples in a grid

Author:
    Jorge Andrés Hernández Galeano
    jhernandezgan@unal.edu.co

Date:
    2023-08-26
    
Usage:
    -Images to plot can be selected either from the WIPs dataset or from the directory of specified images
    -Images 116x256
"""

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

from skimage import  io
from skimage import transform as tf
from skimage.io import imsave, imread

from utils import *


n_rows = 8                   # number of rows in the grid  
n_columns = 4                # number of columns in the grid
samples = n_rows*n_columns   #number of samples to plot


plot_generated = True       #Choose to plot either images from dataset or from path directory
                            # True: plot from directory      #False: plot from dataset


#### Parameters for generated images #####
root_generated = 'Generated_images/ResNet_wsdiv_51'  #Path to the generated images directory                                         #Number of samples to plot
generated_samples_n =50                              #Number of generated samples in total    


##### Parameters for dataset ################
images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'    
category = 15                                        #Choose the species category to plot
dataset = get_dataset(images_root, images_reference,category=category)
n = len(dataset)

# Create a  grid of subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(12,11 ),tight_layout=True)

images = []
for i in range(samples):
    generated_sample = io.imread(os.path.join(root_generated, ".".join([str(random.randint(generated_samples_n)),'jpg']))) if plot_generated else (tf.resize((dataset[random.randint(n)])['image'], (116,256), anti_aliasing= True, preserve_range=True)).astype(np.uint8)
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