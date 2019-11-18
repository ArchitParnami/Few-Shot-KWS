import torch
import torch.nn as nn
import torch.nn.functional as F
from protonets.models.encoder.baseUtil import Flatten, get_padding
from collections import OrderedDict


class TC(nn.Module):
    def __init__(self):
        super(TC, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 3)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, in_height, in_width, out_channels, kernel, dilation, stride=2):
        super(ResidualBlock, self).__init__()
        self.k_size = kernel
        self.stride2 = 1
        self.k_1D = (1,1)
        
        if in_channels != out_channels:
            self.stride1 = stride
            self.upsample = True
        else:
            self.stride1 = 1
            self.upsample = False
       
        self.conv1 = nn.Conv2d(in_channels, out_channels, self.k_size, stride=self.stride1, 
                        bias=False, padding=get_padding(in_height, in_width, self.k_size[0], 
                        self.k_size[1], self.stride1,d_h=dilation), dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, self.k_size, stride=self.stride2, 
                        bias=False,  padding=get_padding(in_height, in_width, self.k_size[0], 
                        self.k_size[1], self.stride2, d_h=dilation), dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, self.k_1D, stride=self.stride1, 
                        bias=False, padding=get_padding(in_height, in_width, self.k_1D[0], 
                        self.k_1D[1], self.stride1))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        self.encoder = nn.Sequential(
            self.conv1, self.bn1, self.relu1,
            self.conv2, self.bn2
        )
        
        self.upsampler = nn.Sequential(
            self.conv3, self.bn3, self.relu3
        )

        self.relu = nn.ReLU()
        

    def forward(self, x):
        a = self.encoder(x)
        if self.upsample:
            b = self.upsampler(x)
        else:
            b = x
        out = a + b
        out = self.relu(out)
        return out


class TCResNet(nn.Module):
    def __init__(self, in_channels, in_height, in_width, n_blocks, n_channels, 
                 conv_kernel, res_kernel, dilation, stride=2):
        super(TCResNet, self).__init__()
        self.conv_k_size = conv_kernel
        self.conv_stride = 1
        self.conv_channels = n_channels[0]

        self.tc = TC()
        self.conv1 = nn.Conv2d(in_channels, self.conv_channels, self.conv_k_size,
                        stride=self.conv_stride, bias=False, padding= get_padding(in_height, 
                        in_width, self.conv_k_size[0], self.conv_k_size[1], self.conv_stride,
                        d_h=dilation[0]), dilation = dilation[0])
       
        self.resnet = self.build_resnet(in_height, in_width, n_blocks, n_channels[1:], 
                                        dilation[1:], res_kernel, stride)
        self.avg_pool = nn.AvgPool2d((in_height, in_width))
        self.flatten = Flatten()

    def build_resnet(self,in_height, in_width, n_blocks, n_channels, dilation, res_kernel, stride):
        res_blocks = []
        for i in range(n_blocks):
            input_channels = self.conv_channels if i == 0 else n_channels[i-1]
            output_channels = n_channels[i]
            res_blocks.append(('Res_{}'.format(i+1),
                                ResidualBlock(input_channels, in_height, in_width, 
                                              output_channels, res_kernel, dilation[i], stride)))
        return nn.Sequential(OrderedDict(res_blocks))

    def forward(self, x):
        out = self.tc(x)
        out = self.conv1(out)
        out = self.resnet(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        return out
    

def TCResNet8(in_c, in_h, in_w, width_multiplier=1.0):
    n_blocks = 3
    n_channels = [16, 24, 32, 48]
    conv_kernel = (3,1)
    res_kernel = (9,1)
    dilation = [1] * 4
    n_channels = [int(x * width_multiplier) for x in n_channels]

    return TCResNet(in_w, in_h, in_c, n_blocks, n_channels, conv_kernel, res_kernel, dilation)


def TCResNet14(in_c, in_h, in_w, width_multiplier=1.0):
    n_blocks = 6
    n_channels = [16, 24, 24, 32, 32, 48, 48]
    conv_kernel = (3,1)
    res_kernel = (9,1)
    dilation = [1] * 4
    n_channels = [int(x * width_multiplier) for x in n_channels]

    return TCResNet(in_w, in_h, in_c, n_blocks, n_channels, conv_kernel, res_kernel, dilation)

def TCResNet8Dilated(in_c, in_h, in_w, width_multiplier=1.0):
    n_blocks = 3
    n_channels = [16, 24, 32, 48]
    n_channels = [int(x * width_multiplier) for x in n_channels]
    conv_kernel = (3,1)
    res_kernel = (7,1)
    dilation = [1,1,2,4]
    return TCResNet(in_w, in_h, in_c, n_blocks, n_channels, conv_kernel, res_kernel, dilation, stride=1)