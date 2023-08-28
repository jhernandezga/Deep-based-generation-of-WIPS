"""
Script generate.py
==================

It generates images from a trained GAN model

Author:
    Jorge Andrés Hernández Galeano
    https://github.com/jhernandezga
    jhernandezga@unal.edu.co

Date:
    2023-08-27

Usage:
    Specify load and saving paths. Specify number of samples to generate, network and network type

"""


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
import torchgan 

import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
# Torchgan Imports
from torchgan.models import DCGANGenerator
from torchgan.models import DCGANDiscriminator
from torchgan.losses import *
from torchgan.trainer import Trainer
#from torchgan.logging import Logger

from Resources.wips_dataset import wipsDataset
from models_param import *
from models_set import *
from skimage.io import imsave

from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))
 
load_path  = 'ResNetExperiments\models\model_14_0\gan9.model'  # Path to load the trained model
saving_path = 'gen'               # Path to save the generated images

samples = 50        # Number of images to generate

#net_type:   0 = AEEE, 1 =DCGAN, 2 = ResNet
net_type = 2


params = torch.load(load_path)
#params_net specify the network on which the model was trained,  models/models_param
params_net = resnet_network["generator"]
state_dict = params['generator']


mapping = [
    {
    'encoder.module.': 'module.encoder.',
    'encoder_fc.module.': 'module.encoder_fc.',
    'decoder_fc.module.': 'module.decoder_fc.',
    'decoder.module.': 'module.decoder.',
},
    {
    'model.module.': 'module.model.',
},
    {
    'model.module.': 'module.model.',
    'linear.module.': 'module.linear.',
    'label_embedding.module':'module.label_embedding',
} ]

state_dict_mapping = mapping[net_type]

new_state_dict = OrderedDict()

for key, value in state_dict.items():
    for old_prefix, new_prefix in state_dict_mapping.items():
        if key.startswith(old_prefix):
            new_key = key.replace(old_prefix, new_prefix)
            new_state_dict[new_key] = value
            break

netGen = ResNetGenerator(**params_net['args']).to(device)
netGen = torch.nn.DataParallel(netGen).to(device)
netGen.load_state_dict(new_state_dict)
netGen.eval()

print('Epoch: ',params['epoch'])

for i in range(samples):
    
    if net_type == 2:
        z = torch.randn(1, 1, 1, 100, device=device)
    elif net_type == 1:
        z = torch.randn(1, 100, 1, 1, device=device)
    else:
        z = torch.randn(1, 256)
    
    with torch.no_grad():
        # Get generated image from the noise vector using
        # the trained generator.
        z = z.to(next(netGen.parameters()).device)
        generated_img = netGen(z).detach().cpu()[0]
        generated_img = transforms.functional.crop(generated_img, top=0, left=0, height=116, width=256)
        generated_img = np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0))
        generated_img = (generated_img * 255).numpy().astype(np.uint8)
        imsave("{}/{}.jpg".format(saving_path,i), generated_img)



