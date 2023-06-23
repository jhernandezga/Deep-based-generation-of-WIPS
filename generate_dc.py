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
from torchgan.trainer import Trainer
#from torchgan.logging import Logger

from Resources.wips_dataset import wipsDataset
from models_param import *
from skimage.io import imsave

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))

 
load_path  = 'model_test4/gan4.model'
saving_path = 'Generated_images/DCGAN_wsgp/3'


params = torch.load(load_path)
params_net = dcgan_network["generator"]

test = False
samples = 20

print('Epoch: ',params['epoch'])

#print(params['generator'])
netGen = DCGANGenerator(**params_net['args']).to(device)
netGen.load_state_dict(params['generator'])
netGen.eval()

if test:
    z = torch.randn(1,100,1,1, device=device)
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
        z = torch.randn(1,100,1,1, device=device)
        with torch.no_grad():
            # Get generated image from the noise vector using
            # the trained generator.
            z = z.to(next(netGen.parameters()).device)
            generated_img = netGen(z).detach().cpu()
            generated_img = transforms.functional.crop(generated_img, top=0, left=0, height=116, width=256)
            generated_img = np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0))
            imsave("{}/{}.jpg".format(saving_path,i), generated_img)