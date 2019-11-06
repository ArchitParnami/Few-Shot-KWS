import torch.nn as nn
import torch


def same_padding(n, f, s):
    p = int((s * (n-1) - n + f) / 2)
    return p    

def get_padding(in_h, in_w, f_h, f_w, s):
    p_h = same_padding(in_h, f_h, s)
    p_w = same_padding(in_w, f_w, s)
    return (p_h, p_w)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class TC(nn.Module):
    def __init__(self):
        super(TC, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 3)