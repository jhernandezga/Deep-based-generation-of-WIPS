"""
Script: calculate_metrics.py
==================

Script to calculate metrics(SSIM and FID) in the dataset or generated images

Author:
    Jorge Andrés Hernández Galeano
    https://github.com/jhernandezga
    jhernandezga@unal.edu.co
    

Date:
    2023-08-27

Usage:
    -Specify the required parameters and choose to calculate metrics over the dataset, the generated images or compare the generated samples with the dataset images
    -Run the script

#Output: Metrics Displayed in terminal output
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

from skimage import metrics, io
from skimage import transform as tf
from skimage.io import imsave, imread

from torchmetrics.image.fid import FrechetInceptionDistance

from utils import *

################### REQUIRED PARAMETERS #################

root_generated = 'Generated_images/ResNet_wsdiv_51'       #Path to generated images

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'

species_category = 51                                      #The species category in the dataset/ specify it dataset images are used for calculations

generated_samples = 50                                     #number of generated samples to take from /root_generated


test_generated = True    #Metrics calculated over dataset or generated images True: Uses Generated images , False: Uses Images from dataset, to compare generated images with datase: False #
FFID_calcul = True       #True: Calculates FID #
SSIM_calcul = True       #True: Calculates SSIM #

SSIM_inter = True        #True: Calculates SSIM between generated images and images from dataset #
#########################################################################################################################






dataset = get_dataset(images_root, images_reference,category=species_category)

n = len(dataset) if not test_generated else generated_samples 

ssim_max = 0
ssim_min = 2
ssim_mean = 0
iterations = 0
index_twin = []

if SSIM_calcul:
    for i in range(n):
        image = (tf.resize((dataset[i])['image'], (116,256), anti_aliasing= True, preserve_range=True)).astype(np.uint8) if not test_generated else io.imread(os.path.join(root_generated, ".".join([str(i),'jpg'])))
        for j in range(i+1, n):
            if i != j: 
                image2 = (tf.resize((dataset[j])['image'], (116,256), anti_aliasing= True, preserve_range=True)).astype(np.uint8) if not test_generated else io.imread(os.path.join(root_generated, ".".join([str(j),'jpg'])))
                ssim =  metrics.structural_similarity(image, image2, channel_axis = 2, data_range=255)
                iterations += 1
                ssim_mean += ssim
                
                if not test_generated and ssim > 0.9:
                    index_twin.append((i, j))
            
                ssim_max = max(ssim_max, ssim)
                ssim_min = min(ssim_min, ssim)
                
                
    ssim_mean = ssim_mean/iterations

    print("SSIM Max: ",ssim_max)
    print("SSIM Min: ",ssim_min)
    print("SSIM Mean: ",ssim_mean)
    print("Twins: ",index_twin)
    
    
if SSIM_inter:
    
    ssim_list = []
    ssim_gen_original_mean = 0
    iter_index = 0
    for i in range(generated_samples):
        image = io.imread(os.path.join(root_generated, ".".join([str(i),'jpg'])))
        for j in range(len(dataset)):
            image2 = (tf.resize((dataset[j])['image'], (116,256), anti_aliasing= True, preserve_range=True)).astype(np.uint8) 
            ssim =  metrics.structural_similarity(image, image2, channel_axis = 2, data_range=255)
            ssim_list.append(ssim)
            
    sim_gen_original_mean = sum(ssim_list) / len(ssim_list)
    ssim_max = max(ssim_list)
    ssim_min = min(ssim_list)
    
    print("SSIM Max Gen-Ori: ",ssim_max)
    print("SSIM Min Gen-Ori: ",ssim_min)
    print("SSIM Mean Gen-Ori: ",sim_gen_original_mean)

    

if FFID_calcul:
    original_dist = []
    generated_dist = []
    
    n = len(dataset)

    for i in range(n):
        original_sample = (dataset[i])['image']
        original_sample = (tf.resize(original_sample, (116,256), anti_aliasing= True, preserve_range=True)).astype(np.uint8)
        original_dist.append(torch.from_numpy(original_sample.transpose(2,0,1)))
        
    for i in range(generated_samples):
        generated_sample = io.imread(os.path.join(root_generated, ".".join([str(i),'jpg'])))
        generated_dist.append(torch.from_numpy(generated_sample.transpose(2,0,1)))

    generated_dist = torch.stack(generated_dist)
    original_dist = torch.stack(original_dist)
    
    fid = FrechetInceptionDistance(feature=64)
    fid.update(original_dist, real=True)
    fid.update(generated_dist, real=False)
    fid_r = fid.compute()
    
    print("FID: ", fid_r.item())