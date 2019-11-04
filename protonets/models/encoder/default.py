import torch
import torch.nn as nn
from protonets.models.encoder.baseUtil import *

class C64(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(C64, self).__init__()
        self.encoder = self.build_encoder(in_channels, hid_channels, out_channels)

    def build_encoder(self, in_dim, hid_dim, z_dim):
        encoder = nn.Sequential(
            self.conv_block(in_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, z_dim),
            Flatten()
        )
        return encoder
            
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        out = self.encoder(x)
        return out