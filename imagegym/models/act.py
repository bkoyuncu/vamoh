import torch
import torch.nn as nn
from imagegym.config import cfg
from imagegym.contrib.act import *
import imagegym.register as register

def sin_activation(input):
    return torch.sin(input)

class SinActivation(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return sin_activation(input) # simply apply already implemented SiLU

act_dict = {
    'relu': nn.ReLU(inplace=cfg.mem.inplace),
    'selu': nn.SELU(inplace=cfg.mem.inplace),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(inplace=cfg.mem.inplace),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1, inplace=cfg.mem.inplace),
    'lrelu_025': nn.LeakyReLU(negative_slope=0.25, inplace=cfg.mem.inplace),
    'lrelu_05': nn.LeakyReLU(negative_slope=0.5, inplace=cfg.mem.inplace),
    'softmax': nn.Softmax(),
    'identity': nn.Identity(),
    "sinus": SinActivation()
}

act_dict = {**register.act_dict, **act_dict}
