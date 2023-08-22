

import matplotlib.pyplot as plt
from numpy import random
import os

from skimage import metrics, io
from skimage import transform as tf
from skimage.io import imsave, imread
        
from Resources.wips_dataset import wipsDataset

from utils import *

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv' 
species_category = 144  


dataset = get_dataset(images_root, images_reference,category=species_category)
n = len(dataset)



dcgan_wsgp = 'Generated_images/DCGAN_wsgp'
resnet_wsgp = 'Generated_images/ResNet_wsgp_s144'
resnet_wsdiv = 'Generated_images/ResNet_wsdiv'
aee_mse = 'Generated_images/AEE_MSE'
aee_perceptual = 'Generated_images/AEE_perceptual_adversarial'
aee_ws_mse = 'Generated_images/AEE_ws_MSE'

nsamples = 5
ncolums = 2
fig, ax = plt.subplots(nrows = nsamples, ncols = 6, figsize=(20,10))
plt.tight_layout()

fontsize = 13
for i in range(nsamples):
    ax[i,0].axis('off')
    ax[i,0].imshow(tf.resize((dataset[random.randint(n)])['image'],(116,256), anti_aliasing= True))  
    
    ax[i,1].axis('off')
    ax[i,1].imshow(io.imread(os.path.join(resnet_wsdiv, ".".join([str(i),'jpg']))))
    
    ax[i,2].axis('off')
    ax[i,2].imshow(io.imread(os.path.join(resnet_wsgp, ".".join([str(i),'jpg']))))
    
    ax[i,3].axis('off')
    ax[i,3].imshow(io.imread(os.path.join(dcgan_wsgp, ".".join([str(i),'jpg']))))
    
    ax[i,4].axis('off')
    ax[i,4].imshow(io.imread(os.path.join(aee_mse, ".".join([str(i),'jpg']))))
    
    ax[i,5].axis('off')
    ax[i,5].imshow(io.imread(os.path.join(aee_perceptual, ".".join([str(i),'jpg']))))
    
    if i == 0:
        ax[i,0].set_title('Original', fontsize= fontsize)
        ax[i,1].set_title('ResNet WS-DIV, FID: 12.44', fontsize= fontsize)
        ax[i,2].set_title('ResNet WS-GP, FID: 13.14', fontsize= fontsize)
        ax[i,3].set_title('DCGAN WS-GP, FID: 20.02', fontsize= fontsize)
        ax[i,4].set_title('AEE MSE Loss, FID: 57.19', fontsize= fontsize)
        ax[i,5].set_title('AEE Perceptual Loss, FID: 37.61', fontsize= fontsize)
  
plt.show()        

fig.savefig('comparison3.png')