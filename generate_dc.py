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

# Set random seed for reproducibility
#manualSeed = 999
#random.seed(manualSeed)
#torch.manual_seed(manualSeed)
#print("Random Seed: ", manualSeed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv' 
load_path  = 'model_test4/gan4.model'


params = torch.load(load_path)
params_net =  {
        "name": DCGANGenerator,
        "args": {
            "out_size":256,
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "batchnorm": True,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh(),
        }}

print('Epoch: ',params['epoch'])

#print(params['generator'])
netGen = DCGANGenerator(**params_net['args']).to(device)
netGen.load_state_dict(params['generator'])
netGen.eval()
#for param_tensor in netGen.state_dict():
#    print(param_tensor, "\t", netGen.state_dict()[param_tensor].size())

#print(netGen)

noise = torch.randn(1,100,1,1, device=device)
print(noise.size())

with torch.no_grad():
	# Get generated image from the noise vector using
	# the trained generator.
    generated_img = netGen(noise).detach().cpu()

plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))

plt.show(block = True)