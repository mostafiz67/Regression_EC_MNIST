"""
Author: Md Mostafizur Rahman
File: CNN design for the MNIST dataset 
Some code for the model is inherited from https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

"""

import os
from torchsummary import summary
from torch import nn
from src.constants import CHECKPOINT, OUT


def get_model():
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(         
                nn.Conv2d(
                    in_channels=1,              
                    out_channels=16,            
                    kernel_size=5,              
                    stride=1,                   
                    padding=2,                  
                ),                              
                nn.ReLU(),                      
                nn.MaxPool2d(kernel_size=2),    
            )
            self.conv2 = nn.Sequential(         
                nn.Conv2d(16, 32, 5, 1, 2),     
                nn.ReLU(),                      
                nn.MaxPool2d(2),                
            )
            # fully connected layer, output 10 classes for classification and 1 class for regression
            self.out = nn.Linear(32 * 7 * 7, 1)
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            x = x.view(x.size(0), -1)       
            output = self.out(x)
            return output, x    # return x for visualization
    return CNN()


if __name__ == "__main__":
    conv_model = get_model()
    summary(conv_model, (1, 28, 28))
    

 
