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

from vgg_perceptual_loss import VGGPerceptualLoss
from vgg_loss import VGGLoss

from math import ceil, log

devices = ["cuda:1", "cuda:2", "cuda:3"]


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