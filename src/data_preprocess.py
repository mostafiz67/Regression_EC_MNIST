"""
Author: Md Mostafizur Rahman
File: Preprocessing MNIST train and test datasets

"""

import torch
from torchvision import datasets
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.utils import make_grid

def data_loaders():
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transforms.ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = transforms.ToTensor()
    )

    return train_data, test_data


    
if __name__ == "__main__":
    print(data_loaders())
