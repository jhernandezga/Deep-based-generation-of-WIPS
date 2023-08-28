"""
Module: models_param.py
==================

It instantiates the models of "models_set.py" using defined parameters for training and definition of the networks.

Author:
    Jorge Andrés Hernández Galeano
    jhernandezga@unal.edu.co
    https://github.com/jhernandezga

Date:
    2023-08-26

Description:
    Networks are usually defined for their generator, discriminator and loss classes. 
    Here an object is instantiated for each of those components of the network, in order to use them
    easily and in a fast way in the training module

    Instantiantion of classes is done following the Torchgan Framework
    https://torchgan.readthedocs.io/en/latest/getting_started/basic_example.html
"""
    
# General Imports
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


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
from torchgan.models import AutoEncodingDiscriminator

from torchgan.losses import *
from models_set import *




######################################  DCGAN   #######################################


# DCGAN Network for 256x256 images
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
            "step_channels": 32,
            "batchnorm": True,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2)
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}


#DCGAN Network for 512x512 images
dcgan_network_2x = {
    "generator": {
        "name": DCGANGenerator,
        "args": {
            "out_size":512,
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
            "in_size":512,
            "in_channels": 3,
            "step_channels": 64,
            "batchnorm": True,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2)
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}

#DCGAN Network for 512x512 images using 64 step channels
dcgan_network_2x_2 = {
    "generator": {
        "name": DCGANGenerator,
        "args": {
            "out_size":512,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 64,
            "batchnorm": True,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": DCGANDiscriminator,
        "args": {
            "in_size":512,
            "in_channels": 3,
            "step_channels": 64,
            "batchnorm": True,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2)
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}


# ResNet network 256x256 images
resnet_network = {
    "generator": {
        "name": ResNetGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ResNetDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
    },
}


#ResNet Network for 256x256 images using leaky ReLU
resnet_network_l = {
    "generator": {
        "name": ResNetGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "last_nonlinearity": nn.Tanh(),
            "leaky": True,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ResNetDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
            "leaky": True,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
    },
}


#ResNet Network modified learning rate parameters
resnet_network_2 = {
    "generator": {
        "name": ResNetGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 200,
            "out_channels": 3,
            "step_channels": 32,
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ResNetDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}


# ResNet network 256x256 images with packed discriminator ->PACGAN
resnet_network_pack = {
    "generator": {
        "name": ResNetGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": PacResNetDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
            "packing_num":2
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}


#ResNet Network for 512X512 images
resnet_network_2x = {
    "generator": {
        "name": ResNetGenerator,
        "args": {
            "out_size":512,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ResNetDiscriminator,
        "args": {
            "in_size":512,
            "in_channels": 3,
            "step_channels": 32,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}


#ResNet Network 256x256 images with spectral normalization
resnet_network_sn = {
    "generator": {
        "name": ResNetGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ResNetDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
            'spectral_normalization': True,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}



#Network combination of ResNet Generator and DCGAN discriminator
hybrid_network = {
    "generator": {
        "name": ResNetGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
     "discriminator": {
        "name": DCGANDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
            "batchnorm": True,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2)
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}


# Modified ResNet network to include DLLE and DisoMap
mod_wsgp_network = {
    "generator": {
        "name": ResNetGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ResNetDiscriminatorMod,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
            "num_outcomes" :8,
            "spectral_normalization": True,
            "new_model": True,
            "use_adaptive_reparam": False
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}

#Another Resnet implementation for testing purposes
resnet_256 = {
    "generator": {
        "name": ResNetGenerator256,
        "args": {
            "encoding_dims": 128,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0, 0.9)}},
    },
    "discriminator": {
        "name": ResNetDiscriminator256,
        "args": {
            "in_channels": 3,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0, 0.9)}},
    },
}


# Conditional ResNet 256x256
c_resnet = {
    "generator": {
        "name": ConditionalResNetGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 128,
            "num_classes": 145,
            "out_channels": 3,
            "step_channels": 32,
            "label_embed_size": 10,
            "last_nonlinearity": nn.Tanh(),
            "leaky": True,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": ConditinalResNetDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
            "num_classes": 145,
            "leaky": True,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
    },
}

minimax_losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
wgangp_losses = [
    WassersteinGeneratorLoss(),
    WassersteinDiscriminatorLoss(),
    WassersteinGradientPenalty(),
]
lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]

wgandiv_losses = [
    WassersteinGeneratorLoss(),
    WassersteinDiscriminatorLoss(),
    WassersteinDivergence(), 
]

pgngan_losses = [
    HingeDiscriminatorLoss(),
    HingeGeneratorLoss(),
]

wgangp_pack_losses = [
    PackedWasserteinGeneratorLoss(),
    PackedWassersteinDiscriminatorLoss(),
    PackedWassersteinGradientPenalty(),
    
]

wsgp_mod_losses = [
    LossDLL(),
    LossEntropyDiscriminator(),
    MaFLoss(),
    DIsoMapLoss(),
    #LossEntropyGenerator(),
    #WassersteinDiscriminatorLoss(),
    WassersteinGradientPenalty(),
    WassersteinGeneratorLoss()
]



###############################      AEE      ##########################################


# Adversarial autoencoder network 256x256
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


############################################################### BEGAN ######################################################################

#BEGAN Network 256x256
began_network = {
    "generator": {
        "name": EncoderGeneratorBEGAN,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "scale_factor": 2,
            "batchnorm": True,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": AutoEncodingDiscriminator,
        "args": {
            "in_size":256,
            "in_channels": 3,
            "step_channels": 32,
            "scale_factor": 2,
            "batchnorm": True,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
}

began_loss = [BoundaryEquilibriumGeneratorLoss(),
            BoundaryEquilibriumDiscriminatorLoss(gamma= 0.5)]