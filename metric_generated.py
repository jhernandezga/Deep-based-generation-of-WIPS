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

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv' 

samples = 100

root_generated = 'Generated_images/Resnet_wsdiv_58_st'
images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'

species_category = 58   

dataset = get_dataset(images_root, images_reference,category=species_category)
n = len(dataset)



original_sample = (dataset[random.randint(n)])['image']

resized_image = tf.resize(original_sample, (116,256), anti_aliasing= True, preserve_range=True) 

plt.imshow(resized_image)
plt.show()

i = 5
generated_sample = io.imread(os.path.join(root_generated, ".".join([str(i),'jpg'])))

plt.imshow(generated_sample)
plt.show()
 
ssim = 0
psnr = 0
mse = 0

original_dist = []
generated_dist = []

for i in range(samples):
    original_sample = (dataset[random.randint(n)])['image']
    original_sample = (tf.resize(original_sample, (116,256), anti_aliasing= True, preserve_range=True)).astype(np.uint8)
    generated_sample = io.imread(os.path.join(root_generated, ".".join([str(i),'jpg'])))
    
   
    generated_dist.append(torch.from_numpy(generated_sample.transpose(2,0,1)))
        
    ssim += metrics.structural_similarity(original_sample, generated_sample, channel_axis = 2)
    psnr += metrics.peak_signal_noise_ratio(original_sample, generated_sample)
    mse += metrics.mean_squared_error(original_sample, generated_sample)
    
ssim = ssim/samples
psnr = psnr/samples
mse = mse/samples

generated_dist = torch.stack(generated_dist)

n = len(dataset)

for i in range(n):
    original_sample = (dataset[i])['image']
    original_sample = (tf.resize(original_sample, (116,256), anti_aliasing= True, preserve_range=True)).astype(np.uint8)
    original_dist.append(torch.from_numpy(original_sample.transpose(2,0,1)))

original_dist = torch.stack(original_dist)

fid = FrechetInceptionDistance(feature=64)
fid.update(original_dist, real=True)
fid.update(generated_dist, real=False)
fid_r = fid.compute()


print('SSIM: {}'.format(ssim))
print('PSNR: {}'.format(psnr))
print('MSE: {}'.format(mse))
print('FID: {}'.format(fid_r.item()))

""" dcgan_wsgp = 'Generated_images/DCGAN_wsgp'
aee_mse = 'Generated_images/AEE_MSE'
aee_perceptual = 'Generated_images/AEE_perceptual_adversarial'
aee_ws_mse = 'Generated_images/AEE_ws_MSE'

nsamples = 5
ncolums = 2
fig, ax = plt.subplots(nrows = nsamples, ncols = 5, figsize=(20,10))
plt.tight_layout()

for i in range(nsamples):
    ax[i,0].axis('off')
    ax[i,0].imshow(tf.resize((dataset[random.randint(n)])['image'],(116,256), anti_aliasing= True))  
    
    ax[i,1].axis('off')
    ax[i,1].imshow(io.imread(os.path.join(dcgan_wsgp, ".".join([str(i),'jpg']))))
    
    ax[i,2].axis('off')
    ax[i,2].imshow(io.imread(os.path.join(aee_mse, ".".join([str(i),'jpg']))))
    
    ax[i,3].axis('off')
    ax[i,3].imshow(io.imread(os.path.join(aee_ws_mse, ".".join([str(i),'jpg']))))
  
    ax[i,4].axis('off')
    ax[i,4].imshow(io.imread(os.path.join(aee_perceptual, ".".join([str(i),'jpg']))))
  
plt.show()        

fig.savefig('comparison.png') """