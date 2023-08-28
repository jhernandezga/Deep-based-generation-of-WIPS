
"""
Module:   models_set
==================

This module contains the internal implementation of the classes of all generative models

Author:
    Jorge Andrés Hernández Galeano
    https://github.com/jhernandezga
    jhernandez@unal.edu.co

Date:
    2023-08-26

Description:

Models are set following the Torchgan Framework conventions. 
https://torchgan.readthedocs.io/en/latest/

- MODELS:  

AdversarialAutoencoderGenerator
AdversarialAutoencoderDiscriminator

EncoderGeneratorBEGAN

ResNetGenerator
ResNetDiscriminator

ConditionalResNetGenerator
ConditinalResNetDiscriminator

PacResNetDiscriminator
ResNetDiscriminatorMod

BigGanGenerator
BigGanDiscriminator

- Losses:

WasserteinAutoencoderGeneratorLoss
WasserteinAutoencoderDiscriminatorLoss
WassersteinGradientPenaltyMod
WasserteinL1AutoencoderGeneratorLoss
AdversarialAutoencoderDiscriminatorLoss
AdversarialAutoencoderGeneratorLoss
PackedWasserteinGeneratorLoss
PackedWassersteinDiscriminatorLoss

WassersteinDivergence
PackedWassersteinGradientPenalty

LossDLL
LossEntropyGenerator
LossEntropyDiscriminator
MaFLoss
DIsoMapLoss

HingeDiscriminatorLoss
HingeGeneratorLoss

"""

# General Imports
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import copy
import datetime
import time
import timeit
import warnings
from collections import OrderedDict

# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import torch.nn as nn
#from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.init as init

import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils


# Torchgan Imports
import torchgan.models as models
from torchgan.losses import DiscriminatorLoss, GeneratorLoss
from torchgan.losses import *



from torchgan.losses.functional import (
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
    wasserstein_gradient_penalty,
)

from models.vgg_perceptual_loss import VGGPerceptualLoss
from models.vgg_loss import VGGLoss

from math import ceil, log

from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding

from models.Blocks import (DiscriminatorBlock, DiscriminatorTop,
                           GSynthesisBlock, InputBlock)
from models.CustomLayers import (EqualizedConv2d, EqualizedLinear,
                                 PixelNormLayer, Truncation)

from models.CrossResplicaBN import ScaledCrossReplicaBatchNorm2d

devices = ["cuda:0", "cuda:1", "cuda:2"]


#####################################################################################################################
#############################                        AEE                        ####################################
#####################################################################################################################

class AdversarialAutoencoderGenerator(models.Generator):
    def __init__(
        self,
        encoding_dims,
        input_size,
        input_channels,
        step_channels=4,
        nonlinearity=nn.LeakyReLU(0.2),
    ):
        super(AdversarialAutoencoderGenerator, self).__init__(encoding_dims)
        encoder = [
            nn.Sequential(
                nn.Conv2d(input_channels, step_channels, 5, 2, 2), nonlinearity
            )
        ]
        size = input_size // 2
        channels = step_channels
        while size > 1:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 4, 5, 4, 2),
                    nn.BatchNorm2d(channels * 4),
                    nonlinearity,
                )
            )
            channels *= 4
            size = size // 4
        self.encoder = nn.Sequential(*encoder)
        self.encoder_fc = nn.Sequential(nn.Linear(
            channels, encoding_dims
        ),nn.Tanh()
                                        
                                        )  # Can add a Tanh nonlinearity if training is unstable as noise prior is Gaussian
        
        
        step_1 = 128
        self.decoder_fc = nn.Sequential(nn.Linear(encoding_dims, step_1), nonlinearity)
        size = 1
        channels = step_1
        
        decoder = [
            nn.Sequential(
                nn.ConvTranspose2d(channels, channels//2, 4, 4), nonlinearity)
        ]
        
        channels //= 2
        size *= 4
        
        while size < input_size // 2:
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1),
                    nn.BatchNorm2d(channels // 2),
                    nonlinearity,
                )
            )
            channels //= 2
            size *= 2
        if size != input_size:    
            decoder.append(nn.ConvTranspose2d(channels, input_channels, 4, 2, 1))
        else:
            decoder.append(nn.ConvTranspose2d(channels, input_channels, 5, 1, 2))
            
        self.decoder = nn.Sequential(*decoder)
        
    def sample(self, noise):
        noise = self.decoder_fc(noise)
        noise = noise.view(-1, noise.size(1), 1, 1)
        return self.decoder(noise)

    def forward(self, x):
        if self.training:
            encoding = self.encoder(x)
            encoding = self.encoder_fc(
                encoding.view(
                    -1, encoding.size(1) * encoding.size(2) * encoding.size(3)
                    )            )
            #generated images  from decoder and
            #latent vectors from encoder
            return self.sample(encoding), encoding
        else:
            return self.sample(x)
        
class AdversarialAutoencoderDiscriminator(models.Discriminator):
    def __init__(self, input_dims, nonlinearity=nn.LeakyReLU(0.2)):
        super(AdversarialAutoencoderDiscriminator, self).__init__(input_dims)
        model = [nn.Sequential(nn.Linear(input_dims, input_dims // 2), nonlinearity)]
        size = input_dims // 2
        while size > 16:
            model.append(
                nn.Sequential(
                    nn.Linear(size, size // 2), nn.BatchNorm1d(size // 2), nonlinearity
                )
            )
            size = size // 2
        model.append(nn.Linear(size, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class WasserteinAutoencoderGeneratorLoss(GeneratorLoss):
    def __init__(self , reduction="mean", override_train_ops=None):
        super(WasserteinAutoencoderGeneratorLoss, self).__init__(reduction, override_train_ops)
        #self.vgg_perceptual =  VGGPerceptualLoss()
        self.vgg_perceptual =  VGGLoss()
        self.vgg_perceptual.to(devices[0])
    
    def forward(self, real_inputs, gen_inputs, dgz):
        loss1 = self.vgg_perceptual(gen_inputs, real_inputs)
        loss2 = wasserstein_generator_loss(dgz)
        loss = 0.5*loss1 + 0.5*loss2
        return loss

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        recon, encodings = generator(real_inputs)
        optimizer_generator.zero_grad()
        
        #classification of latent vectors from encoder
        dgz = discriminator(encodings)
        
        #loss, comparisom between real and generated images, also discrinator loss
        loss = self.forward(real_inputs, recon, dgz)
        loss.backward()
        optimizer_generator.step()
        return loss.item()


class WasserteinAutoencoderDiscriminatorLoss(DiscriminatorLoss):
    def forward(self, dx, dgz):
        return wasserstein_discriminator_loss(dx, dgz)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        _, encodings = generator(real_inputs)
        noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
        optimizer_discriminator.zero_grad()
        dx = discriminator(noise)
        dgz = discriminator(encodings)
        loss = self.forward(dx, dgz)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()

class WassersteinGradientPenaltyMod(DiscriminatorLoss):
    def __init__(self, reduction="mean", lambd=10.0, override_train_ops=None):
        super(WassersteinGradientPenaltyMod, self).__init__(
            reduction, override_train_ops
        )
        self.lambd = lambd
        self.override_train_ops = override_train_ops

    def forward(self, interpolate, d_interpolate):
        return wasserstein_gradient_penalty(
            interpolate, d_interpolate, self.reduction
        )

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        
        batch_size = real_inputs.size(0)
        noise = torch.randn(
                batch_size, generator.encoding_dims, device=device
            )
        _, encodings = generator(real_inputs)
        
        optimizer_discriminator.zero_grad()
        
        eps = torch.rand(1).item()
        interpolate = eps * noise + (1 - eps) * encodings
        
        d_interpolate = discriminator(interpolate)
        loss = self.forward(interpolate, d_interpolate)
        weighted_loss = self.lambd * loss
        weighted_loss.backward()
        optimizer_discriminator.step()
        return loss.item()


class WasserteinL1AutoencoderGeneratorLoss(GeneratorLoss):
    def __init__(self , reduction="mean", override_train_ops=None):
        super(WasserteinL1AutoencoderGeneratorLoss, self).__init__(reduction, override_train_ops)
        #self.vgg_perceptual =  VGGPerceptualLoss()
        self.vgg_perceptual =  VGGLoss()
        self.vgg_perceptual.to(devices[0])
        self.L1 = nn.L1Loss()
    
    def forward(self, real_inputs, gen_inputs, dgz):
        loss1 = self.vgg_perceptual(gen_inputs, real_inputs)
        loss2 = wasserstein_generator_loss(dgz)
        loss3 = self.L1(gen_inputs, real_inputs)
        loss = 0.4*loss1 +0.4*loss3+ 0.2*loss2
        return loss

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        recon, encodings = generator(real_inputs)
        optimizer_generator.zero_grad()
        
        #classification of latent vectors from encoder
        dgz = discriminator(encodings)
        
        #loss, comparisom between real and generated images, also discrinator loss
        loss = self.forward(real_inputs, recon, dgz)
        loss.backward()
        optimizer_generator.step()
        return loss.item()
    


class AdversarialAutoencoderDiscriminatorLoss(DiscriminatorLoss):
    def forward(self, dx, dgz):
        target_real = torch.ones_like(dx)
        target_fake = torch.zeros_like(dx)
        loss = 0.5 * F.binary_cross_entropy_with_logits(dx, target_real)
        loss += 0.5 * F.binary_cross_entropy_with_logits(dgz, target_fake)
        return loss

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        _, encodings = generator(real_inputs)
        noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
        optimizer_discriminator.zero_grad()
        dx = discriminator(noise)
        dgz = discriminator(encodings)
        loss = self.forward(dx, dgz)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()
    


##############  Adversarial +  MSE
    
class AdversarialAutoencoderGeneratorLoss(GeneratorLoss):
    def forward(self, real_inputs, gen_inputs, dgz):
        loss = 0.5 * F.mse_loss(gen_inputs, real_inputs)
        target = torch.ones_like(dgz)
        loss += 0.5 * F.binary_cross_entropy_with_logits(dgz, target)
        return loss

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        recon, encodings = generator(real_inputs)
        optimizer_generator.zero_grad()
        
        #classification of latent vectors from encoder
        dgz = discriminator(encodings)
        
        #loss, comparisom between real and generated images, also discrinator loss
        loss = self.forward(real_inputs, recon, dgz)
        loss.backward()
        optimizer_generator.step()
        return loss.item()
    


##############  AUTOENCODER BEGAN #############
class EncoderGeneratorBEGAN(models.Generator):
    def __init__(
        self,
        encoding_dims=100,
        out_size=32,
        out_channels=3,
        step_channels=64,
        scale_factor=2,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="none",
    ):
        super(EncoderGeneratorBEGAN, self).__init__(encoding_dims, label_type)
        if out_size < (scale_factor ** 4) or ceil(log(out_size, scale_factor)) != log(
            out_size, scale_factor
        ):
            raise Exception(
                "Target image size must be at least {} and a perfect power of {}".format(
                    scale_factor ** 4, scale_factor
                )
            )
        num_repeats = int(log(out_size, scale_factor)) - 3
        same_filters = scale_factor + 1
        same_pad = scale_factor // 2
        if scale_factor == 2:
            upsample_filters = 3
            upsample_stride = 2
            upsample_pad = 1
            upsample_output_pad = 1
        else:
            upsample_filters = scale_factor
            upsample_stride = scale_factor
            upsample_pad = 0
            upsample_output_pad = 0
        self.ch = out_channels
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.ELU() if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        init_dim = scale_factor ** 3
        self.init_dim = init_dim

        if batchnorm is True:
            self.fc = nn.Sequential(
                nn.Linear(self.encoding_dims, (init_dim ** 2) * self.n),
                nn.BatchNorm1d((init_dim ** 2) * self.n),
            )
            initial_unit = nn.Sequential(
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n),
                nl,
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n),
                nl,
            )
            upsample_unit = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n),
                nl,
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n),
                nl,
            )
            
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.encoding_dims, (init_dim ** 2) * self.n)
            )
            initial_unit = nn.Sequential(
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nl,
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nl,
            )
            upsample_unit = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nl,
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nl,
            )

        last_unit = nn.Sequential(
            nn.Conv2d(self.n, self.ch, same_filters, 1, same_pad, bias=True), last_nl
        )
        model = [initial_unit]
        for i in range(num_repeats):
            model.append(upsample_unit)
            out_size = out_size // scale_factor
        model.append(last_unit)
        self.model = nn.Sequential(*model)
        self._weight_initializer()
    
    def forward(self, z):
        r"""Calculates the output tensor on passing the encoding ``z`` through the Generator.

        Args:
            z (torch.Tensor): A 2D torch tensor of the encoding sampled from a probability
                distribution.

        Returns:
            A 4D torch.Tensor of the generated image.
        """
        x = self.fc(z)
        x = x.view(-1, self.n, self.init_dim, self.init_dim)
        return self.model(x)
    
    
    
################################### WGANGP -ResNet #########################################

class FirstResBlockDiscriminator(nn.Module):
    """ This is the first discriminator block as in the paper """
    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)



class ResidualBlock(nn.Module):
    def __init__(self,input_dim, output_dim, resample ,kernel_size = 3, size = 256, spectral_normalization = False, new_model = False, leaky = False):
        super(ResidualBlock, self).__init__()
        if resample == 'up':
            self.shortcut = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor = 2),
                nn.Conv2d(in_channels= input_dim, out_channels = output_dim, kernel_size = kernel_size, padding = 'same')
                #nn.Conv2d(in_channels= input_dim, out_channels = output_dim, kernel_size = 1, padding = 'same')  
            )
            self.output = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor = 2),
                nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = kernel_size, padding = 'same'),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                nn.Conv2d(in_channels = output_dim, out_channels = output_dim, kernel_size = kernel_size, padding = 'same'),
                )
                
        elif resample == 'down':
            
            if spectral_normalization:
                
                if new_model:
                    self.shortcut = nn.Sequential(
                        nn.AvgPool2d(kernel_size = 2),
                        spectral_norm(nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = 1, padding = 'same')),
                    )
                    self.output = nn.Sequential(
                        nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                        spectral_norm(nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = kernel_size, padding = 'same')),
                        nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                        spectral_norm(nn.Conv2d(in_channels = output_dim, out_channels = output_dim, kernel_size = kernel_size, padding = 'same')),
                        nn.AvgPool2d(kernel_size = 2),      
                    )    
                
                
                else:
                    self.shortcut = nn.Sequential(
                        nn.AvgPool2d(kernel_size = 2),
                        spectral_norm(nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = 1, padding = 'same')),
                        nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                    )
                    self.output = nn.Sequential(
                        spectral_norm(nn.Conv2d(in_channels = input_dim, out_channels = input_dim, kernel_size = kernel_size, padding = 'same')),
                        nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                        nn.AvgPool2d(kernel_size = 2),
                        nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                        spectral_norm(nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = kernel_size, padding = 'same')),         
                    )    
            
            else:
            
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size = 2),
                    nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = 1, padding = 'same'),
                    nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                )
                self.output = nn.Sequential(
                    nn.LayerNorm(normalized_shape = [input_dim, size, size]),
                    nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                    nn.Conv2d(in_channels = input_dim, out_channels = input_dim, kernel_size = kernel_size, padding = 'same'),
                    nn.LayerNorm(normalized_shape = [input_dim, size, size]),
                    nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                    nn.AvgPool2d(kernel_size = 2),
                    nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = kernel_size, padding = 'same'),         
                )
            
        else:
            raise Exception('invalid resample value')


    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.output(x)
        out = identity + residual
        return out
    
class ResidualBlockConv(nn.Module):
    def __init__(self,input_dim, output_dim, resample ,kernel_size = 4, size = 256, spectral_normalization = False, new_model = False):
        super(ResidualBlockConv, self).__init__()
        if resample == 'up':
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels= input_dim, out_channels = output_dim, kernel_size = kernel_size, stride=2, padding = 1)
                #nn.Conv2d(in_channels= input_dim, out_channels = output_dim, kernel_size = 1, padding = 'same')  
            )
            self.output = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels= input_dim, out_channels = output_dim, kernel_size = kernel_size, stride=2, padding = 1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
                nn.Conv2d(in_channels = output_dim, out_channels = output_dim, kernel_size = kernel_size, padding = 'same'),
                )
                
        elif resample == 'down':
            
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = kernel_size, stride=2, padding = 1),
                )
                self.output = nn.Sequential(
                    nn.LayerNorm(normalized_shape = [input_dim, size, size]),
                    nn.ReLU(),
                    nn.Conv2d(in_channels = input_dim, out_channels = input_dim, kernel_size = kernel_size, padding = 'same'),
                    nn.LayerNorm(normalized_shape = [input_dim, size, size]),
                    nn.ReLU(),
                    nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = kernel_size, stride=2, padding = 1),         
                )
            
        else:
            raise Exception('invalid resample value')


    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.output(x)
        out = identity + residual
        return out



class ResNetGenerator(models.Generator):
    def __init__(
        self,
        encoding_dims=100,
        out_size=256,
        out_channels=3,
        step_channels=64,
        scale_factor=2,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="none",
        leaky = False
    ):
        super(ResNetGenerator, self).__init__(encoding_dims, label_type)
        
        num_repeats = out_size.bit_length() - 4
        self.n = step_channels
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        
        d = int(self.n * (2 ** num_repeats))
        
        self.init_channels = int(self.n * (2 ** num_repeats))
        
        model = []
        
        self.linear = nn.Linear(encoding_dims, 4*4*d) 
        
        for i in range(num_repeats):
            model.append(
                ResidualBlock(d,d//2, resample = 'up', leaky = leaky)
                #ResidualBlockConv(d,d//2, resample = 'up')          
            )
            d = d//2
        
        model.append(
            ResidualBlock(d,d//2, resample = 'up', leaky= leaky)
            #ResidualBlockConv(d,d//2, resample = 'up')
        )
        d = d//2
        
        model.append(
           nn.Sequential( nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Conv2d(in_channels = d, out_channels = out_channels, kernel_size = 3, padding = 'same'),
            last_nl )       
        )
        
        self.model = nn.Sequential(*model)
        self._weight_initializer()
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1,self.init_channels,4,4)
        return self.model(x)

    def _weight_initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)





class ResNetDiscriminator(models.Discriminator):
    def __init__(
        self,
        in_size = 256 ,
        in_channels=3,
        step_channels=64,
        kernel_size = 3,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="none",
        spectral_normalization = False,
        leaky = False
    ):
        super(ResNetDiscriminator, self).__init__(in_channels, label_type)
        
        num_repeats = in_size.bit_length() - 4
        self.n = step_channels
        size = in_size
        d = self.n
        model = []
        
        model.append(
            nn.Sequential(
               nn.Conv2d(in_channels = in_channels, out_channels = d, kernel_size = kernel_size, padding = 'same'))
            )
        
        for i in range(num_repeats):
            model.append(
                nn.Sequential(
                        ResidualBlock(input_dim = d, output_dim = d*2, resample = 'down', size = size, spectral_normalization = spectral_normalization, leaky=leaky)
                    
                ) 
            )
            d *= 2
            size = size//2
    
        model.append(
            nn.Sequential(
                nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features = int(d*size**2), out_features = 1)) 
            )
        
        self.model = nn.Sequential(*model)
        self._weight_initializer()
        
    def forward(self, x):
        return self.model(x)
    
    def _weight_initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)


class ConditionalResNetGenerator(models.Generator):
    def __init__(
        self,
        encoding_dims=100,
        out_size=256,
        out_channels=3,
        step_channels=64,
        num_classes = 100,
        label_embed_size = 100,
        scale_factor=2,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="required",
        leaky = False
    ):
        super(ConditionalResNetGenerator, self).__init__(encoding_dims, label_type)
        
        num_repeats = out_size.bit_length() - 4
        self.n = step_channels
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        self.num_classes = num_classes
        d = int(self.n * (2 ** num_repeats))
        
        self.init_channels = int(self.n * (2 ** num_repeats))
        
        model = []
        
        self.label_embedding = nn.Embedding(num_classes, label_embed_size)
        self.linear = nn.Linear(encoding_dims+label_embed_size, 4*4*d) 
        
        for i in range(num_repeats):
            model.append(
                ResidualBlock(d,d//2, resample = 'up', leaky = leaky)
                #ResidualBlockConv(d,d//2, resample = 'up')          
            )
            d = d//2
        
        model.append(
            ResidualBlock(d,d//2, resample = 'up', leaky= leaky)
            #ResidualBlockConv(d,d//2, resample = 'up')
        )
        d = d//2
        
        model.append(
           nn.Sequential( nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Conv2d(in_channels = d, out_channels = out_channels, kernel_size = 3, padding = 'same'),
            last_nl )       
        )
        
        self.model = nn.Sequential(*model)
        self._weight_initializer()
    
    def forward(self, x, label):
        x = x.reshape([x.shape[0], -1])
        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape([label.shape[0], -1])
        c = torch.cat((x, label_embed), dim=1)
        
        x = self.linear(c)
        y = x.view(-1,self.init_channels,4,4)
        return self.model(y)

    def _weight_initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)


class ConditinalResNetDiscriminator(models.Discriminator):
    def __init__(
        self,
        in_size = 256 ,
        in_channels=3,
        step_channels=64,
        kernel_size = 3,
        num_classes = 100,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="required",
        spectral_normalization = False,
        leaky = False
    ):
        super(ConditinalResNetDiscriminator, self).__init__(in_channels, label_type)
        
        self.num_classes = num_classes
        num_repeats = in_size.bit_length() - 4
        self.n = step_channels
        size = in_size
        self.img_size = in_size
        d = self.n
        model = []
        
        self.label_embedding = nn.Embedding(num_classes, self.img_size*self.img_size)
        
        model.append(
            nn.Sequential(
               nn.Conv2d(in_channels = in_channels+1, out_channels = d, kernel_size = kernel_size, padding = 'same'))
            )
        
        for i in range(num_repeats):
            model.append(
                nn.Sequential(
                        ResidualBlock(input_dim = d, output_dim = d*2, resample = 'down', size = size, spectral_normalization = spectral_normalization, leaky=leaky)
                    
                ) 
            )
            d *= 2
            size = size//2
    
        model.append(
            nn.Sequential(
                nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features = int(d*size**2), out_features = 1)) 
            )
        
        self.model = nn.Sequential(*model)
        self._weight_initializer()
        
    def forward(self, x, label):
        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape([label_embed.shape[0], 1, self.img_size, self.img_size])
        x = torch.cat((x, label_embed), dim=1)
        return self.model(x)
    
    def _weight_initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)
















class PacResNetDiscriminator(models.Discriminator):
    def __init__(
        self,
        in_size = 256 ,
        in_channels=3,
        step_channels=64,
        kernel_size = 3,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="none",
        spectral_normalization = False,
        packing_num = 2,
        leaky = True
    ):
        super(PacResNetDiscriminator, self).__init__(in_channels, label_type)
        
        num_repeats = in_size.bit_length() - 4
        self.n = step_channels
        size = in_size
        d = self.n
        model = []
        
        model.append(
            nn.Sequential(
               nn.Conv2d(in_channels = in_channels*packing_num, out_channels = d, kernel_size = kernel_size, padding = 'same'))
            )
        
        for i in range(num_repeats):
            model.append(
                nn.Sequential(
                        ResidualBlock(input_dim = d, output_dim = d*2, resample = 'down', size = size, spectral_normalization = spectral_normalization)
                    
                ) 
            )
            d *= 2
            size = size//2
    
        model.append(
            nn.Sequential(
                nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features = int(d*size**2), out_features = 1)) 
            )
        
        self.model = nn.Sequential(*model)
        self._weight_initializer()
        
    def forward(self, x):
        return self.model(x)
    
    def _weight_initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)

class PackedWasserteinGeneratorLoss(GeneratorLoss):
    
    def __init__(self, reduction="mean", override_train_ops=None, packing_num = 2):
        super(PackedWasserteinGeneratorLoss, self).__init__(reduction, override_train_ops)
        self.packing_num = packing_num
        
    def forward(self, fgz):
        return wasserstein_generator_loss(fgz, self.reduction)
    
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        device,
        batch_size,
        labels=None,
    ):
        optimizer_generator.zero_grad()
        generated_images = []
        for i in range(self.packing_num):
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            generated_images.append(generator(noise))
        generated_images = torch.cat(generated_images, dim = 1)
        
        dgz = discriminator(generated_images)
        loss = self.forward(dgz)
        loss.backward()
        optimizer_generator.step()
        
        return loss.item()
        

class PackedWassersteinDiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, reduction="mean", clip=None, override_train_ops=None, packing_num = 2):
        super(PackedWassersteinDiscriminatorLoss, self).__init__(
            reduction, override_train_ops
        )
        if (isinstance(clip, tuple) or isinstance(clip, list)) and len(clip) > 1:
            self.clip = clip
        else:
            self.clip = None
        self.packing_num = packing_num

    def forward(self, fx, fgz):
        return wasserstein_discriminator_loss(fx, fgz, self.reduction)
    
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
            
        batch_size = real_inputs.size(0)
        optimizer_discriminator.zero_grad()
        
        generated_images = []
        for i in range(self.packing_num):
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            generated_images.append(generator(noise))
        generated_images = torch.cat(generated_images, dim = 1)
        dx = discriminator(real_inputs)
        dgz = discriminator(generated_images)
        
        loss = self.forward(dx, dgz)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()


class WassersteinDivergence(DiscriminatorLoss):
    def __init__(self, reduction="mean", k=2, override_train_ops=None, p = 6):
        super(WassersteinDivergence, self).__init__(reduction, override_train_ops)
        self.k = k
        self.p = p

    def forward(self, real_gradient_norm, fake_gradient_norm):
        return torch.mean(real_gradient_norm + fake_gradient_norm) * self.k / 2
    
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        
        if labels is None and (
                generator.label_type == "required"
                or discriminator.label_type == "required"
            ):
                raise Exception("GAN model requires labels for training")
        
            
        batch_size = real_inputs.size(0)
        
        real_inputs = torch.autograd.Variable(real_inputs.type(torch.Tensor), requires_grad=True)
        real_inputs = real_inputs.to(device)
                
        optimizer_discriminator.zero_grad()
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
       
        if generator.label_type == "none":
            generated_images = generator(noise)
        elif generator.label_type == "required":
            generated_images = generator(noise, labels)
        
        real_grad_outputs = torch.full((real_inputs.size(0),1), 1, dtype=torch.float32, requires_grad=False, device=device)
        fake_grad_outputs = torch.full((generated_images.size(0),1), 1, dtype=torch.float32, requires_grad=False, device=device)
        
        if discriminator.label_type == "none":
            real_outputs =  discriminator(real_inputs)
            fake_outputs = discriminator(generated_images)
        elif discriminator.label_type == "required":
            real_outputs =  discriminator(real_inputs, labels)
            fake_outputs = discriminator(generated_images, labels)
         
        #real_outputs =  torch.autograd.Variable(real_outputs, requires_grad=True)
        
        
        real_gradient = torch.autograd.grad(
        outputs=real_outputs,
        inputs=real_inputs,
        grad_outputs=real_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
             )[0]
        
        fake_gradient = torch.autograd.grad(
        outputs=fake_outputs,
        inputs=generated_images,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
            )[0]
        
        real_gradient_norm = real_gradient.view(real_gradient.size(0), -1).pow(2).sum(1) ** (self.p / 2)
        fake_gradient_norm = fake_gradient.view(fake_gradient.size(0), -1).pow(2).sum(1) ** (self.p / 2)

        loss = self.forward(real_gradient_norm, fake_gradient_norm)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()

    
class PackedWassersteinGradientPenalty(DiscriminatorLoss):
    def __init__(self, reduction="mean", lambd=10.0, override_train_ops=None, packing_num = 2):
        super(PackedWassersteinGradientPenalty, self).__init__(reduction, override_train_ops)
        self.lambd = lambd
        self.override_train_ops = override_train_ops
        self.packing_num = packing_num


    def forward(self, interpolate, d_interpolate):
        return wasserstein_gradient_penalty(interpolate, d_interpolate, self.reduction)
    
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
            
        batch_size = real_inputs.size(0)
        optimizer_discriminator.zero_grad()
        
        generated_images = []
        for i in range(self.packing_num):
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            generated_images.append(generator(noise))
        generated_images = torch.cat(generated_images, dim = 1)
        
        eps = torch.rand(1).item()
        interpolate = eps * real_inputs + (1 - eps) * generated_images
        
        d_interpolate = discriminator(interpolate)

        loss = self.forward(interpolate, d_interpolate)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()


class ResNetDiscriminatorMod(models.Discriminator):
    def __init__(
        self,
        in_size = 256 ,
        in_channels=3,
        step_channels=64,
        kernel_size = 3,
        num_outcomes = 8,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="none",
        spectral_normalization = False,
        use_adaptive_reparam = True,
        new_model = False
    ):
        super(ResNetDiscriminatorMod, self).__init__(in_channels, label_type)
        
        num_repeats = in_size.bit_length() - 4
        self.use_adaptive_reparam = use_adaptive_reparam
        self.num_outcomes = num_outcomes
        self.n = step_channels
        size = in_size
        d = self.n
        model = []
        
        model.append(
            nn.Sequential(
                #nn.LayerNorm(normalized_shape = [in_channels, size, size]),
                spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = d, kernel_size = kernel_size, padding = 'same')))
            )
        
        for i in range(num_repeats):
            model.append(
                nn.Sequential(
                        ResidualBlock(input_dim = d, output_dim = d*2, resample = 'down', size = size, spectral_normalization = spectral_normalization, new_model = new_model)
                    
                ) 
            )
            d *= 2
            size = size//2
        
        model.append(
            nn.Sequential(
                nn.ReLU(),
                #nn.LayerNorm(normalized_shape = [d, size, size]),
                ) 
            )
        self.model = nn.Sequential(*model)
        
        
        self.flat_model = nn.Flatten()
       
        
        input_size = int(d*size**2)
        out_size = num_outcomes
        
        fc = nn.Linear(input_size, out_size, bias=False)
        nn.init.orthogonal_(fc.weight.data)
        self.fc = spectral_norm(fc)
        
        #resampling trick
        self.reparam = spectral_norm(nn.Linear(in_features = input_size, out_features = 2*num_outcomes, bias = False))
        nn.init.orthogonal_(self.reparam.weight.data)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self._weight_initializer()
       
    def forward(self, x):
        y = self.model(x)
        y = self.flat_model(y)
        output = self.fc(y).view(-1, self.num_outcomes)
         
        if self.use_adaptive_reparam:
            stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
            mu, logvar = stat_tuple.chunk(2, 1)
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn(x.shape[0], self.num_outcomes, 1, 1).to(stat_tuple)
            output = epsilon.mul(std).add_(mu).view(-1, self.num_outcomes)
        return output
    
    def _weight_initializer(self):
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)
    
class LossDLL(DiscriminatorLoss):
    
    def data_aug(self,tensor):
        dim = random.choice([2,3])
        tensor = torch.flip(tensor,[dim])
        tensor = tensor + 0.001 * torch.randn_like(tensor)
        return tensor

    def forward(self, t_z, t_z_aug, f_z, f_z_aug):
        
        return 0.5*F.mse_loss(t_z,t_z_aug.detach()) + 0.5*F.mse_loss(f_z,f_z_aug.detach())
           
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        batch_size=None,
        labels=None,
    ):
        noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
        optimizer_discriminator.zero_grad()
        
        generated_img = generator(noise)
        
        
        realdata_aug = self.data_aug(real_inputs)
        generated_img_aug = self.data_aug(generated_img)
        t_z = discriminator(real_inputs)
        f_z = discriminator(generated_img)
        t_z_aug = discriminator(realdata_aug)
        f_z_aug = discriminator(generated_img_aug)
        
        loss = self.forward(t_z, t_z_aug, f_z, f_z_aug)
        loss.backward(retain_graph=True)
        optimizer_discriminator.step()
        return loss.item()



class LossEntropyGenerator(GeneratorLoss):
    def __init__(self , reduction="mean", override_train_ops=None, data_length = 100, batch_size = 8):
        super(LossEntropyGenerator, self).__init__(reduction, override_train_ops)
        self.replay_buffer_generate = self.entropy_buffer()
        self.len_data = data_length
        self.epoch = 1
        self.batch_number = 0 
        self.batch_size = 8
    
    def forward(self,batch_code,replay_buffer, batch_size ):
        batch_code_r = batch_code.unsqueeze(0)
        dist_metric = torch.mean(torch.pow((replay_buffer-batch_code_r),2),dim=2)

        dist_min,idx = torch.min(dist_metric,dim=0)

        loss = torch.mean(dist_min)

        batch_code_s = torch.stack([batch_code.detach()],dim=0)

        replay_buffer = torch.cat((replay_buffer[batch_size:],batch_code_s),dim=0)

        return loss,replay_buffer
    
    def entropy_buffer(self, buffer_size = 1024):
        outcomes = 8
        replay_buffer = torch.FloatTensor(buffer_size, outcomes).uniform_(-1, 1)
        replay_buffer = torch.stack([replay_buffer.detach() for i in range(8)], dim = 1)
        return replay_buffer
    
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        self.batch_number += 1
        temp = self.len_data//batch_size
        
        if self.batch_number == temp:
            self.epoch += 1
            
        self.replay_buffer_generate = self.replay_buffer_generate.to(device)
        noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
        entropy_lr = 1
        optimizer_generator.zero_grad()
        
        generated_img = generator(noise)
        f_z = discriminator(generated_img)
        
        
        if self.epoch < 100:
            lr = self.epoch/100. * entropy_lr
        else:
            lr =  1.0 * entropy_lr 
        
        loss_entropy, self.replay_buffer_generate  = self.forward(f_z,self.replay_buffer_generate, batch_size)
        loss =  -lr * torch.log(loss_entropy)
        
        loss.backward(retain_graph=True)
        optimizer_generator.step()
        return loss.item()

class LossEntropyDiscriminator(DiscriminatorLoss):
    
    def __init__(self , reduction="mean", override_train_ops=None, data_length = 100, batch_size = 8):
        super(LossEntropyDiscriminator, self).__init__(reduction, override_train_ops)
        self.replay_buffer_real = self.entropy_buffer()
        self.len_data = data_length
        self.epoch = 1
        self.batch_number = 0
        self.batch_size = batch_size
        
    
    def forward(self,batch_code,replay_buffer, batch_size ):
        batch_code_r = batch_code.unsqueeze(0)
        dist_metric = torch.mean(torch.pow((replay_buffer-batch_code_r),2),dim=2)

        dist_min,idx = torch.min(dist_metric,dim=0)

        loss = torch.mean(dist_min)

        batch_code_s = torch.stack([batch_code.detach()],dim=0)

        replay_buffer = torch.cat((replay_buffer[batch_size:],batch_code_s),dim=0)

        return loss,replay_buffer

    
    def entropy_buffer(self, buffer_size = 1024):
        outcomes = 8
        replay_buffer = torch.FloatTensor(buffer_size, outcomes).uniform_(-1, 1)
        replay_buffer = torch.stack([replay_buffer.detach() for i in range(8)], dim = 1)
        return replay_buffer
    
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        
        optimizer_discriminator.zero_grad()
        self.batch_number += 1
        temp = self.len_data//batch_size
        
        if self.batch_number == temp:
            self.epoch += 1
        
        
        self.replay_buffer_real = self.replay_buffer_real.to(device)
        entropy_lr = 1
        real_inputs = real_inputs
        t_z = discriminator(real_inputs)
        
        
        if self.epoch < 100:
            lr = self.epoch/100. * entropy_lr
        else:
            lr =  1.0 * entropy_lr
            
        loss_entropy, self.replay_buffer_real = self.forward(t_z,self.replay_buffer_real, batch_size )
        loss =  -lr * torch.log(loss_entropy)
        loss.backward(retain_graph=True)
        optimizer_discriminator.step()
        
        return loss.item()
 
        
class MaFLoss(DiscriminatorLoss):
    def forward(self,t_z,f_z):
        return (-torch.mean(t_z) + torch.mean(f_z))
    
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
        optimizer_discriminator.zero_grad()
        
        generated_img = generator(noise)
    
        t_z = discriminator(real_inputs)
        f_z = discriminator(generated_img)
        loss = self.forward(t_z,f_z)
        loss.backward(retain_graph=True)
        optimizer_discriminator.step()
        
        return loss.item()

class DIsoMapLoss(DiscriminatorLoss):
    
    def forward(self, c_mix, z_mix):
        return F.mse_loss(c_mix,z_mix)
    
    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
        optimizer_discriminator.zero_grad()
        
        generated_img = generator(noise)
    
        t_z = discriminator(real_inputs)
        f_z = discriminator(generated_img)
        alpha = torch.randn(real_inputs.size(0),1).to(device)
        mix_z = alpha * t_z + f_z * (1-alpha)
        mix_z = mix_z + torch.normal(mean = 0., std = 0.05, size= (mix_z.size())).to(device)
        
        alpha = alpha.view(-1,1,1,1)
        
        mix_input = alpha * real_inputs + (1-alpha) * generated_img
        loss = self.forward(discriminator(mix_input),mix_z.detach())
        loss.backward(retain_graph=True)
        optimizer_discriminator.step()
                
        return loss.item()


##### New attempts RESNET #####

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leaky = False):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )
        # residual
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    
class OptimizedDisblock(nn.Module):
    def __init__(self, in_channels, out_channels, leaky = False):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        # residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False, leaky=False):
        super().__init__()
        # shortcut
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)
        # residual
        residual = [
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))

class ResNetDiscriminator256(models.Discriminator):
    def __init__(
        self,
        in_channels=3,
        label_type="none",
        leaky = True
    ):
        super(ResNetDiscriminator256, self).__init__(in_channels)
            
        self.model = nn.Sequential(
            OptimizedDisblock(3, 64,leaky=leaky),
            DisBlock(64, 128, down=True,leaky=leaky),
            DisBlock(128, 256, down=True,leaky=leaky),
            DisBlock(256, 512, down=True,leaky=leaky),
            DisBlock(512, 512, down=True,leaky=leaky),
            DisBlock(512, 1024, down=True,leaky=leaky),
            DisBlock(1024, 1024, leaky=leaky),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(1024, 1)
        # initialize weight
        self.initialize()
               
    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

class ResNetGenerator256(models.Generator):
    def __init__(
        self,
        in_channels=3,
        label_type="none",
        encoding_dims=100,
        leaky = True
    ):
        super(ResNetGenerator256, self).__init__(encoding_dims)
        self.linear = nn.Linear(encoding_dims, 4 * 4 * 1024)

        self.blocks = nn.Sequential(
            GenBlock(1024, 1024,leaky=leaky),
            GenBlock(1024, 512, leaky=leaky),
            GenBlock(512, 512, leaky=leaky),
            GenBlock(512, 256, leaky=leaky),
            GenBlock(256, 128, leaky=leaky),
            GenBlock(128, 64, leaky=leaky),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 1024, 4, 4)
        return self.output(self.blocks(inputs))


def penalty_normalize_gradient(net_D, x, **kwargs):
    """
                          1 - f
    f_hat = -------------------------------
               ||1 - grad_f ||+ |1 - f|
    """
    x.requires_grad_(True)
    f = net_D(x, **kwargs)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(1 - grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = ((1 - f) / (grad_norm + torch.abs(1 - f)))
    return f_hat


def hinge_loss( pred_real, pred_fake=None):
    if pred_fake is not None:
        loss_real = F.relu(1 - pred_real).mean()
        loss_fake = F.relu(1 + pred_fake).mean()
        loss = loss_real + loss_fake
        return loss
    else:
        loss = -pred_real.mean()
        return loss

class HingeDiscriminatorLoss(DiscriminatorLoss):
    def forward(self, fx, fgz):
        return hinge_loss(fx, fgz)
    
    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs, device, labels=None):
        batch_size = real_inputs.size(0)
        optimizer_discriminator.zero_grad()
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
        generated_images = generator(noise)

        real_fake = torch.cat([real_inputs, generated_images], dim=0)
        
        pred = penalty_normalize_gradient(discriminator, real_fake)
        
        pred_real, pred_fake = torch.split(pred, [real_inputs.shape[0], generated_images.shape[0]], dim=0)
        
        loss = self.forward(pred_real, pred_fake)
        
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()
    
class HingeGeneratorLoss(GeneratorLoss):
    
    def forward(self, fgz):
        return hinge_loss(fgz)
    
    def train_ops(self, generator, discriminator, optimizer_generator, device, batch_size, labels=None):
        optimizer_generator.zero_grad()
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
        generated_images = generator(noise)
        pred_fake = penalty_normalize_gradient(discriminator, generated_images)
        loss = self.forward(pred_fake)
        loss.backward()
        optimizer_generator.step()
        return loss.item()
    



###########BIGGAN##############    
class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Embedding(num_classes, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
    
        embed = self.embed(class_id)
        # print('embed', embed.size())
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        # print(beta.size())
        out = gamma * out + beta

        return out   
    


class GBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, n_class=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False, num_classes = 145):
        super().__init__()

        gain = 2 ** 0.5

        self.conv0 = spectral_norm(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))
        self.conv1 = spectral_norm(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channel, out_channel,
                                                   1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, num_classes)
            self.HyperBN_1 = ConditionalNorm(out_channel, num_classes)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            # print('condition',condition.size()) #condition torch.Size([4, 148])
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        if self.upsample:
            # TODO different form papers
            out = F.upsample(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=F.relu):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

        self.init_conv(self.query_conv)
        self.init_conv(self.key_conv)
        self.init_conv(self.value_conv)
    
    def init_conv(self,conv, glu=True):
        init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()

        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

    
class BigGanGenerator(models.Generator):
    def __init__(self,encoding_dims, label_type="required",num_classes = 145, step_channels = 64):
        super(BigGanGenerator, self).__init__(encoding_dims, label_type)
        
        self.embed = nn.Embedding(num_classes, encoding_dims)

        self.first_view = 16 * step_channels

        self.G_linear = spectral_norm(nn.Linear(20, 4 * 4 * 16 * step_channels))

        self.conv = nn.ModuleList([GBlock(16*step_channels, 16*step_channels, n_class=num_classes),
                                GBlock(16*step_channels, 8*step_channels, n_class=num_classes),
                                GBlock(8*step_channels, 4*step_channels, n_class=num_classes),
                                GBlock(4*step_channels, 2*step_channels, n_class=num_classes),
                                SelfAttention(2*step_channels),
                                GBlock(2*step_channels, 1*step_channels, n_class=num_classes)])

        # TODO impl ScaledCrossReplicaBatchNorm 
        self.ScaledCrossReplicaBN = ScaledCrossReplicaBatchNorm2d(1*step_channels)
        self.colorize = spectral_norm(nn.Conv2d(1*step_channels, 3, [3, 3], padding=1))
        
    def forward(self, input, class_id):
        codes = torch.split(input, 20, 1)
        class_emb = self.embed(class_id)

        out = self.G_linear(codes[0])
        # out = out.view(-1, 1536, 4, 4)
        out = out.view(-1, self.first_view, 4, 4)
        ids = 1
        for i, conv in enumerate(self.conv):
            if isinstance(conv, GBlock):
                
                conv_code = codes[ids]
                ids = ids+1
                condition = torch.cat([conv_code, class_emb], 1)
                # print('condition',condition.size()) #torch.Size([4, 148])
                out = conv(out, condition)

            else:
                out = conv(out)

        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)

        return F.tanh(out)

class BigGanDiscriminator(models.Discriminator):
    def __init__(self, in_channels, step_channels = 64,label_type="required", num_classes = 145):
        super(BigGanDiscriminator, self).__init__(in_channels, label_type)

        def conv(in_channel, out_channel, downsample=True):
            return GBlock(in_channel, out_channel,
                          bn=False,
                          upsample=False, downsample=downsample)
        
        self.pre_conv = nn.Sequential(spectral_norm(nn.Conv2d(3, 1*step_channels, 3,padding=1),),
                                      nn.ReLU(),
                                      spectral_norm(nn.Conv2d(1*step_channels, 1*step_channels, 3,padding=1),),
                                      nn.AvgPool2d(2))
        self.pre_skip = spectral_norm(nn.Conv2d(3, 1*step_channels, 1))

        self.conv = nn.Sequential(conv(1*step_channels, 1*step_channels, downsample=True),
                                  SelfAttention(1*step_channels),
                                  conv(1*step_channels, 2*step_channels, downsample=True),    
                                  conv(2*step_channels, 4*step_channels, downsample=True),
                                  conv(4*step_channels, 8*step_channels, downsample=True),
                                  conv(8*step_channels, 16*step_channels, downsample=True),
                                  conv(16*step_channels, 16*step_channels, downsample=False))

        self.linear = spectral_norm(nn.Linear(16*step_channels, 1))

        self.embed = nn.Embedding(num_classes, 16*step_channels)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)
    
    def forward(self, input, class_id):
        
        out = self.pre_conv(input)
        out = out + self.pre_skip(F.avg_pool2d(input, 2))
        # print(out.size())
        out = self.conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        embed = self.embed(class_id)

        prod = (out * embed).sum(1)

        return out_linear + prod
















