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

import torchgan.models as models
from aeegan_wassertein import AdversarialAutoencoderGenerator, WasserteinAutoencoderDiscriminatorLoss, WasserteinAutoencoderGeneratorLoss, WassersteinGradientPenaltyMod
from models.vgg_loss import VGGLoss
import torchvision.transforms as transforms

from collections import OrderedDict
from models_set import *
from models_param import *
#devices = ["cuda:1", "cuda:2", "cuda:3"] 
#print(torch.cuda.get_device_name(device))

from skimage.io import imsave, imread


load_path  = 'AEE_experiments/models/model_mse/gan4.model'
params = torch.load(load_path, map_location='cpu')
saving_path = 'Generated_images/AEE_MSE'

test = False
samples = 100

print('Epoch: ',params['epoch'])

print(params.keys())
state_dict = params['generator']

devices = ["cuda:0", "cuda:1"] 

""" # create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace(".module", "") # removing ‘.moldule’ from key
    new_state_dict[name] = v
#load params """

state_dict_mapping = {
    'encoder.module.': 'module.encoder.',
    'encoder_fc.module.': 'module.encoder_fc.',
    'decoder_fc.module.': 'module.decoder_fc.',
    'decoder.module.': 'module.decoder.',
}

new_state_dict = OrderedDict()

for key, value in state_dict.items():
    for old_prefix, new_prefix in state_dict_mapping.items():
        if key.startswith(old_prefix):
            new_key = key.replace(old_prefix, new_prefix)
            new_state_dict[new_key] = value
            break

params_net = aee_network['generator']

netGen = AdversarialAutoencoderGenerator(**params_net['args'])
netGen = torch.nn.DataParallel(netGen, devices).to(devices[0])
netGen.load_state_dict(new_state_dict)
netGen.eval()

#print(netGen)

if test:
    z = torch.randn(1,256)
    with torch.no_grad():
        z = z.to(next(netGen.parameters()).device)
        generated_img = netGen(z).detach().cpu()
        generated_img = transforms.functional.crop(generated_img, top=0, left=0, height=116, width=256)
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
        plt.show(block = True)


else:
    for i in range(samples):
        z = torch.randn(1,256)
        with torch.no_grad():
            # Get generated image from the noise vector using
            # the trained generator.
            z = z.to(next(netGen.parameters()).device)
            generated_img = netGen(z).detach().cpu()
            generated_img = transforms.functional.crop(generated_img, top=0, left=0, height=116, width=256)
            generated_img = np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)).cpu().numpy()
            
            image_numpy_uint = (generated_img * 255).astype(np.uint8)
            imsave("{}/{}.jpg".format(saving_path,i), image_numpy_uint)
