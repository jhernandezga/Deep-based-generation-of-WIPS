import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import _t_sne

import pandas as pd
import seaborn as sns
from skimage import transform as tf
import torch
from skimage import  io

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from utils import get_dataset, get_dataloader

#https://danielmuellerkomorowska.com/2021/01/05/introduction-to-t-sne-in-python-with-scikit-learn/
images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv' 

samples = 100
#root_generated = 'Generated_images/ResNet_wsdiv'
root_generated = 'Generated_images/ResNet_wsdiv'
root_generated2 = 'Generated_images/ResNet_wsgp_s144_2'
images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'


species_category = 144

dataset = get_dataset(images_root, images_reference,category=species_category)
#dataset2 = get_dataset(images_root, images_reference,category=46)

data_loader = get_dataloader(images_root, images_reference, batch_size = len(dataset), category = species_category, drop_last = False)
#data_loader2 = get_dataloader(images_root, images_reference, batch_size = len(dataset2), category = 46, drop_last = False)

model = models.resnet18(pretrained=True)
#model = models.vgg16(pretrained = True)
model.eval()

# Remove last classification layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

features = []
for images, _ in data_loader:
    with torch.no_grad():
        features_batch = feature_extractor(images)
        features_batch = features_batch.view(features_batch.size(0), -1).numpy()
        features.extend(features_batch)
""" features2 = []
for images, _ in data_loader2:
    with torch.no_grad():
        features_batch = feature_extractor(images)
        features_batch = features_batch.view(features_batch.size(0), -1).numpy()
        features2.extend(features_batch)
 """
features_gen = []
for i in range(samples):
    with torch.no_grad():
        generated_sample = io.imread(os.path.join(root_generated, ".".join([str(i),'jpg'])))
        generated_sample = torch.from_numpy(generated_sample).permute(2,0,1).unsqueeze(0).to(torch.float32)
        featuresx = feature_extractor(generated_sample)
        featuresx = featuresx.view(featuresx.size(0), -1).numpy()
        features_gen.extend(featuresx)

features_gen2 = []
for i in range(100):
    with torch.no_grad():
        generated_sample = io.imread(os.path.join(root_generated2, ".".join([str(i),'jpg'])))
        generated_sample = torch.from_numpy(generated_sample).permute(2,0,1).unsqueeze(0).to(torch.float32)
        featuresx1 = feature_extractor(generated_sample)
        featuresx1 = featuresx1.view(featuresx1.size(0), -1).numpy()
        features_gen2.extend(featuresx1)  

        
features = np.array(features)
#features2 = np.array(features2)
features_gen = np.array(features_gen)
features_gen2 = np.array(features_gen2)

#normalization
features = (features - features.mean(axis=0)) / features.std(axis=0)
#features2 = (features2 - features2.mean(axis=0)) / features2.std(axis=0)
features_gen = (features_gen - features_gen.mean(axis=0)) / features_gen.std(axis=0)
features_gen2 = (features_gen2 - features_gen2.mean(axis=0)) / features_gen2.std(axis=0)

features = np.nan_to_num(features, nan=0.0)
#features2 = np.nan_to_num(features2, nan=0.0)
features_gen = np.nan_to_num(features_gen, nan=0.0)
features_gen2 = np.nan_to_num(features_gen2, nan=0.0)

#print("NaN values in features_gen trans:", np.isnan(features_gen2).any())

data = np.vstack((features,features_gen, features_gen2))
#data = np.vstack((features, features2, features_gen,features_gen2))
tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=600000)

embedded_features = tsne.fit_transform(data)

plt.figure(figsize=(10, 8))
plt.scatter(embedded_features[:len(features), 0], embedded_features[:len(features), 1], c='b', marker='*')
#plt.scatter(embedded_features[len(features):(len(features)+len(features2)), 0], embedded_features[len(features):(len(features)+len(features2)), 1], c='r', marker='.')
plt.scatter(embedded_features[(len(features)):(len(features)+len(features_gen)), 0], embedded_features[(len(features)):(len(features)+len(features_gen)), 1], c='g', marker='.')
plt.scatter(embedded_features[(len(features)+len(features_gen)):(len(features)+len(features_gen)+len(features_gen2)), 0], embedded_features[(len(features)+len(features_gen)):(len(features)+len(features_gen)+len(features_gen2)), 1], c='r', marker='.')
#plt.scatter(embedded_features[(len(features)+len(features2)+len(features_gen)):(len(features)+len(features2)+len(features_gen)+len(features_gen2)), 0], embedded_features[(len(features)+len(features2)+len(features_gen)):(len(features)+len(features2)+len(features_gen)+len(features_gen2)), 1], c='y', marker='.')
plt.title("t-SNE Visualization of Image Features")
plt.legend(['Real 144', 'Generated 144'])
plt.show()

