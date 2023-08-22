import torch
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader

loader = ModelLoader(
    base_dir = 'StyleGanExperiments/models',   # path to where you invoked the command line tool
    name = 'default'                   # the project name, defaults to 'default'
)

noise   = torch.randn(1, 512).cuda() # noise
styles  = loader.noise_to_styles(noise, trunc_psi = 0.7)  # pass through mapping network
images  = loader.styles_to_images(styles) # call the generator on intermediate style vectors

save_image(images, './sample.jpg') # save your images, or do whatever you desire