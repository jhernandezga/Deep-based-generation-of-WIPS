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

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)


devices = ["cuda:0","cuda:1"]

# Training images
images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'  

images_network = 'DCGAN_experiments/logs/tensorboard/training_network'

#logger directory
train_log_dir = 'ResNetExperiments/logs/train_res_augmix_0'
#checkpoint of saved models
checkpoints_path = 'ResNetExperiments/models/model_res_augmix_0/gan'
#path of generated images
images_path  = 'ResNetExperiments/images/images_res_58_augmix_0'

#Checkpoint load path

load_path = 'ResNetExperiments/models/model_c4/gan9.model'

#Category of species to train
species_category = 58

batch_size = 32
#dont change, modify model   
#Number of generated images at each training epoch
generated_samples = 8 

epochs = 1000

trainer = None 


####################################
###  dcgan_network, aee_network , dcgan_network_2x, began_network, resnet_network_2x, hybrid_network, mod_wsgp_network ###
#resnet_network_pack
network = resnet_network
###########################################################
## DCGAN: minimax_losses, wgangp_losses, lsgan_losses, 
## AEE: wassertein_losses, perceptual_losses, wassL1_losses
losses_net = wgandiv_losses

transform = transforms.AugMix()
#transform = transforms.RandomHorizontalFlip()

if torch.cuda.device_count() > 1:
    # Use deterministic cudnn algorithms
    #torch.backends.cudnn.deterministic = True
    trainer = ParallelTrainer(
    network,losses_net, sample_size = generated_samples, epochs=epochs, devices=devices, log_dir = train_log_dir,  checkpoints= checkpoints_path, recon=images_path, retain_checkpoints = 10, ncritic=10)
else :
    device = torch.device(devices[0] if torch.cuda.is_available() else "cpu")    
    if torch.cuda.is_available():
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
    trainer = Trainer(network,losses_net, sample_size = generated_samples, epochs=epochs, device=device, log_dir = train_log_dir,  checkpoints= checkpoints_path, recon=images_path)


print("CUDA available: ",torch.cuda.is_available())
#print("Device: {}".format(torch.cuda.get_device_name(device)))
print("Epochs: {}".format(epochs))

train_dataloader = get_dataloader(images_reference= images_reference, images_root=images_root,category = species_category,batch_size=batch_size, drop_last= True, transform = transform)
#train_dataloader = get_packed_dataloader(images_reference= images_reference, images_root=images_root,category = species_category,batch_size=batch_size, drop_last=False, packing_num=2)

#trainer.load_model(load_path=load_path)
trainer(train_dataloader)

#for batch_idx, (dataa, target) in enumerate(train_dataloader):
#print(dataa.shape)