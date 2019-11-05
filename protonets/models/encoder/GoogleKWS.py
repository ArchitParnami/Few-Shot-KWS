import torch
import torch.nn as nn
from protonets.models.encoder.baseUtil import Flatten

class cnn_trad_fpool3(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(cnn_trad_fpool3, self).__init__()
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hid_channels,
                    kernel_size=(20,8),stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3)))
        
        conv2  = nn.Sequential(
            nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, 
                    kernel_size=(10,4), stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        lin = nn.Sequential(
            nn.Linear(11776, 32),
            nn.ReLU()
        )

        dnn = nn.Sequential(
            nn.Linear(32, 128),
            nn.Sigmoid()
        )

        self.encoder = nn.Sequential(
            conv1,
            conv2,
            Flatten(),
            lin,
            dnn
        )
    
    def forward(self, x):
        out = self.encoder(x)
        return out


class cnn_trad_fpool3_simple(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(cnn_trad_fpool3_simple, self).__init__()
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hid_channels,
                    kernel_size=(20,8),stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3)))
        
        conv2  = nn.Sequential(
            nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, 
                    kernel_size=(10,4), stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.encoder = nn.Sequential(
            conv1,
            conv2,
            Flatten(),
        )
    
    def forward(self, x):
        out = self.encoder(x)
        return out