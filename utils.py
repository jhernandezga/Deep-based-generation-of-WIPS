# Pytorch and Torchvision Imports
import torch
import torch.nn as nn

import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image


from Resources.wips_dataset import wipsDataset

images_root = 'Resources/Images'
images_reference = 'Resources/wips_reference.csv'        

mean_data = [0.4952,0.3626,0.1201]
std_data = [0.3231,0.2541,0.1689]

#transforms.Normalize(mean=mean_data, std=std_data)
#transform = transforms.Compose(transforms.ToTensor())

# -1 get all dataset
def get_dataset(images_root = images_root,images_reference = images_reference,category = -1):
    dataset = wipsDataset(images_reference,images_root, category = category)
    return dataset


def custom_collate(batch):
    resized_batch = []
    categories = []
    for sample in batch:
        image = sample['image']
        pil_image = to_pil_image(image)  # Convert to PIL image
        resized_image = transforms.Resize((232, 512))(pil_image)
        #resized_image = transforms.Resize((116, 256))(pil_image)
        padded_image = transforms.Pad((0,0,0,280))(resized_image)
        #padded_image = transforms.Pad((0,0,0,24))(resized_image)
        #padded_image = transforms.Pad((0,0,0,140))(resized_image)
        resized_tensor = transforms.ToTensor()(padded_image)  # Convert back to tensor
        resized_batch.append(resized_tensor)
        categories.append(torch.tensor(sample['category']))
        
    #stacked_batch_images = torch.tensor(resized_batch)
    #stacked_categories = torch.tensor(categories)

    stacked_batch_images = torch.stack(resized_batch)
    stacked_categories = torch.stack(categories)

    return stacked_batch_images, stacked_categories

def get_dataloader(images_root = images_root, images_reference = images_reference ,batch_size = 16,train__size_factor = 1,category = -1):
    dataset = get_dataset(images_root = images_root, images_reference = images_reference, category = category)
    train_size = int(train__size_factor * len(dataset))
    train_dataloader = data.DataLoader(dataset, batch_size, shuffle= True, collate_fn= custom_collate)
    return train_dataloader