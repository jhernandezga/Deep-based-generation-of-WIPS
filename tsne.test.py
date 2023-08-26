# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import os

import matplotlib.pyplot as plt
import numpy as np

import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from numpy import random

from Resources.wips_dataset import wipsDataset

from skimage import metrics, io
from skimage import transform as tf
from skimage.io import imsave, imread

from torchmetrics.image.fid import FrechetInceptionDistance

from utils import *

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv' 

samples = 100


categories = []

root_generated_list = ['Generated_images/ResNet_wsdiv_{}'.format(category) for category in categories]


images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'

datasets = [get_dataset(images_root, images_reference,category=category) for category in categories]