"""
Script: training.py
==================

Executes the training procedure in the defined network and parameters

Author:
    Jorge Andrés Hernández Galeano
    https://github.com/jhernandezga
    jhernandezga@unal.edu.co

Date:
    2023-08-26

Description:
    - It uses the training procedure following the Torchgan Framework
    - This module supports multi-GPU training
    - GPU training is highly recommended

Usage:
    Set the paths to the training and the parameters. Run script
    
    If visualization of training at real-time is desired:
        after running this script, execute in a terminal:    tensorboard --logdir = train_log_dir

"""

try:
    import torchgan

    print(f"Existing TorchGAN {torchgan.__version__} installation found")
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchgan"])
    import torchgan

    print(f"Installed TorchGAN {torchgan.__version__}")
    
# General Imports
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
#from IPython.display import HTML

# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import torch.nn as nn

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
from torchgan.trainer import ParallelTrainer, BaseTrainer
#from torchgan.logging import Logger

from Resources.wips_dataset import wipsDataset
from utils import get_dataloader, get_packed_dataloader
from models_param import *


# Training images
images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv' 


torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)



##################################### REQUIRED PARAMETERS ################################################################

## Specify GPU devices to use
devices = ["cuda:0","cuda:1"]

train_log_dir = 'ResNetExperiments\logs'             #Path to logger directory(Where data used for the logger will be saved)

checkpoints_path = 'ResNetExperiments\models\gan'    #Path where checkpoint models will be saved

images_path  = 'ResNetExperiments\images'           #Path where generated images during training will be saved

load_checkpoint = False                                      # False: Training will be started from scratch , True: Training will be started from a checkpoint model
load_path = 'ResNetExperiments/models/model_54_0/gan9.model' #Path to checkpoint model

#Category of species to train
species_category = 54

batch_size = 4
  
generated_samples = 8                      #Number of generated images at each training epoch, used for logging purposes

epochs = 8500                              #Number of training epochs

retain_checkpoints = 10                    #Number of checkpoints to retain X. Last X models will be saved, when #training epochs>X, checkpoints are overwritten
n_critic = 5                               #Number of steps the discriminator will be trained before training the generator
 
"""
Specify the training network and its corresponding loss function

Available pre-instantiated models in models/models_param.py:

->dcgan_network, dcgan_network_2x, hybrid_network
->c_resnet       :conditional ResNet
->resnet_network, resnet_network_l, resnet_network_2x, resnet_network_sn
    - minimax_losses
    - wgangp_losses
    - lsgan_losses
    - wgandiv_losses
->resnet_network_pack          =PACGAN(it requires to use get_packed_dataloader from utils)
    - wgangp_pack_losses
->mod_wsgp_network
    -wsgp_mod_losses

->aee_network
    -wassertein_losses
    -perceptual_losses
    -wassL1_losses
-> began_network
    -began_loss  
"""
network = resnet_network
losses_net = wgandiv_losses

###########################################################################################################################################################################################







# Transformations during training
transform1 = transforms.RandomApply(
    torch.nn.ModuleList([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        ]),
    p=0.5)

transform = transforms.Compose([
        transform1,
        transforms.ToTensor()
        ])

trainer = None
multi_gpu = False
if torch.cuda.device_count() > 1:
    # Use deterministic cudnn algorithms
    #torch.backends.cudnn.deterministic = True
    trainer = ParallelTrainer(
    network,losses_net, sample_size = generated_samples, epochs=epochs, devices=devices, log_dir = train_log_dir,  checkpoints= checkpoints_path, recon=images_path, retain_checkpoints = retain_checkpoints, ncritic=n_critic)
    multi_gpu = True
else :
    device = torch.device(devices[0] if torch.cuda.is_available() else "cpu")    
    if torch.cuda.is_available():
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
    trainer = Trainer(network,losses_net, sample_size = generated_samples, epochs=epochs, device=device, log_dir = train_log_dir,  checkpoints= checkpoints_path, recon=images_path, retain_checkpoints=retain_checkpoints,ncritic=n_critic)


print("CUDA available: ",torch.cuda.is_available())
print("Multi-GPU training: {}".format(multi_gpu))
print("Epochs: {}".format(epochs))

## replace get_dataloader by get_packed_dataloader if PACGAN is being trained
train_dataloader = get_dataloader(images_reference= images_reference, images_root=images_root,category = species_category,batch_size=batch_size, drop_last= False, transform = transform)

if load_checkpoint:
    trainer.load_model(load_path=load_path)
trainer(train_dataloader)
