"""
Script: tsne.py
==================

It performs t-sne dimensional reduction on images from dataset and generated ones, and plot its 2d representation
It uses ResNet18 as feature extractor


Author:
    Jorge Andrés Hernández Galeano
    jhernandezga@unal.edu.co

Date:
    2023-08-27

Usage:
    Set required parameters and run the script
"""
#https://danielmuellerkomorowska.com/2021/01/05/introduction-to-t-sne-in-python-with-scikit-learn/

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import _t_sne

import pandas as pd
from skimage import transform as tf
import torch
from skimage import  io

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from utils import get_dataset, get_dataloader



####################REQUIRED PARAMETERS##################
root_generated = 'Generated_images\ResNet_wsdiv_46'  #Path to generated images

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'


species_category = 46                               #Species category to evaluate
samples = 90                                        #number of generated sample to consider from /root_generated


n_iterations = 2000                     #Number of iterations for tsne algorithm
perplexity = 30                         #Perplexity value for tsne algorithm
random_state = 42                       #Random initial state for tsne algorithm - used for reproducibility 
############################################################

dataset = get_dataset(images_root, images_reference,category=species_category)
data_loader = get_dataloader(images_root, images_reference, batch_size = len(dataset), category = species_category, drop_last = False)

#Set Feature extractor
model = models.resnet18(pretrained=True)
model.eval()

# Remove last classification layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])


#Extract features from original/dataset images
features = []
for images, _ in data_loader:
    with torch.no_grad():
        features_batch = feature_extractor(images)
        features_batch = features_batch.view(features_batch.size(0), -1).numpy()
        features.extend(features_batch)

#Extract features from generated images
features_gen = []
for i in range(samples):
    with torch.no_grad():
        generated_sample = io.imread(os.path.join(root_generated, ".".join([str(i),'jpg'])))
        generated_sample = torch.from_numpy(generated_sample).permute(2,0,1).unsqueeze(0).to(torch.float32)
        featuresx = feature_extractor(generated_sample)
        featuresx = featuresx.view(featuresx.size(0), -1).numpy()
        features_gen.extend(featuresx)
  
        
features = np.array(features)
features_gen = np.array(features_gen)

#Join features in a single array
joined_features = np.vstack((features,features_gen))

#Normalize feature values
joined_features = (joined_features - joined_features.mean(axis=0)) / joined_features.std(axis=0)
joined_features = np.nan_to_num(joined_features, nan=0.0)
data = joined_features

#Set t-sne algorithm
tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, n_iter=n_iterations)

# execute tsne over features
embedded_features = tsne.fit_transform(data)
print(tsne.kl_divergence_)


#plotting
plt.figure(figsize=(10, 8))
plt.scatter(embedded_features[:len(features), 0], embedded_features[:len(features), 1], c='b', marker='*')
plt.scatter(embedded_features[(len(features)):(len(features)+len(features_gen)), 0], embedded_features[(len(features)):(len(features)+len(features_gen)), 1], c='g', marker='.')
plt.title("t-SNE Visualization of Image Features")
plt.legend(['Original images label {}'.format(species_category), 'Generated images label {}'.format(species_category)], loc='upper right',fontsize=18, handlelength=2)
plt.show()

