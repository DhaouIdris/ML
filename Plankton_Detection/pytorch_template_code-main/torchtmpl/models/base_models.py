# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch.nn as nn


def Linear(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    layers = [
        nn.Flatten(start_dim=1),
        nn.Linear(reduce(operator.mul, input_size, 1), num_classes),
    ]
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.enc1 = self.conv_block(in_channels, 64)  
        self.pool1 = nn.MaxPool2d(2)                  

        self.enc2 = self.conv_block(64, 128)          
        self.pool2 = nn.MaxPool2d(2)                  

        self.enc3 = self.conv_block(128, 256)       
        self.pool3 = nn.MaxPool2d(2)                 

        self.enc4 = self.conv_block(256, 512)         
        self.pool4 = nn.MaxPool2d(2)                  

        self.bottleneck = self.conv_block(512, 1024)  

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  
        self.dec4 = self.conv_block(1024, 512)                                 

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.dec3 = self.conv_block(512, 256)                                 

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   
        self.dec2 = self.conv_block(256, 128)                                 

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    
        self.dec1 = self.conv_block(128, 64)                                   

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)             

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        out = self.out_conv(dec1)
        return out

    @staticmethod
    def conv_block(in_channels, out_channels):
        """Convolutional block: 2 convolutions + ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.3)
        )
