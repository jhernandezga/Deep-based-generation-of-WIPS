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
from IPython.display import HTML

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
from models_set import AdversarialAutoencoderDiscriminatorLoss, AdversarialAutoencoderGenerator, AdversarialAutoencoderDiscriminator, AdversarialAutoencoderGeneratorLoss, WassersteinGradientPenaltyMod, WasserteinAutoencoderDiscriminatorLoss, WasserteinAutoencoderGeneratorLoss, WasserteinL1AutoencoderGeneratorLoss

######################################  DCGAN   #######################################

dcgan_network = {
    "generator": {
        "name": DCGANGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "batchnorm": True,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": DCGANDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 64,
            "batchnorm": True,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2)
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}


minimax_losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
wgangp_losses = [
    WassersteinGeneratorLoss(),
    WassersteinDiscriminatorLoss(),
    WassersteinGradientPenalty(),
]
lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]



###############################      AEE      ##########################################

aee_network = {
    "generator": {
        "name": AdversarialAutoencoderGenerator,
        "args": {"encoding_dims": 256, "input_size": 256, "input_channels": 3},
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": AdversarialAutoencoderDiscriminator,
        "args": {"input_dims": 256,},
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}   


wassertein_losses = [
    WasserteinAutoencoderGeneratorLoss(),
    WasserteinAutoencoderDiscriminatorLoss(),
    WassersteinGradientPenaltyMod(),
]

### VGG + Adversarial
perceptual_losses = [
    AdversarialAutoencoderGeneratorLoss(),
    AdversarialAutoencoderDiscriminatorLoss(),
]


#### VGG + L1 + wassertein 
wassL1_losses = [
    WasserteinL1AutoencoderGeneratorLoss(),
    WasserteinAutoencoderDiscriminatorLoss(),
    WassersteinGradientPenaltyMod(),
]


