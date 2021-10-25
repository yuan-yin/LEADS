import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm as sn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from functools import partial
from torch import optim
import math
import numbers
import functools
from scipy import ndimage
from torch.nn import Parameter


#################################################################################
# Credit for Swish                                                              #
#                                                                               #
# Projet: https://github.com/rtqichen/residual-flows                            #
# Copyright (c) 2019 Ricky Tian Qi Chen                                         #
# Licence (MIT): https://github.com/rtqichen/residual-flows/blob/master/LICENSE #
#################################################################################

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))
    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

class LinearEstimator(nn.Module):
    def __init__(self, in_c, out_c, factor=1.0):
        super().__init__()
        self.factor = factor
        self.net = nn.Linear(in_c, out_c, bias=False)

    def forward(self, x):
        return self.net(x) * self.factor

class MLPEstimator(nn.Module):
    def __init__(self, in_c, out_c, hidden, factor):
        super().__init__()
        bias = True
        self.net = nn.Sequential(
            nn.Linear(in_c, hidden, bias=bias),
            Swish(),
            nn.Linear(hidden, hidden, bias=bias),
            Swish(),
            nn.Linear(hidden, hidden, bias=bias),
            Swish(),
            nn.Linear(hidden, out_c, bias=bias),
        )
        self.factor = factor
    
    def forward(self, x):
        flatten = x.dim() == 3
        if flatten:
            batch_size, nc, T = x.shape
            x = x.transpose(2, 1)
        x = self.net(x) * self.factor
        if flatten:
            batch_size, nc, T = x.shape
            x = x.transpose(2, 1)
        return x

#####################################################################################
# Credit for SpectralConv2d_fast, FNO2d                                             #
#                                                                                   #
# Projet: https://github.com/zongyi-li/fourier_neural_operator                      #
# Copyright (c) 2020 Zongyi Li                                                      #
# Licence: https://github.com/zongyi-li/fourier_neural_operator/blob/master/LICENSE #
#####################################################################################
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # self.scale = (1 / (in_channels * out_channels))
        self.scale = 1 / math.sqrt(in_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width)
        # input channel is 3: (w(t, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0, self.bn1, self.bn2, self.bn3 = [nn.Identity() for _ in range(4)]
        self.a0 = Swish()
        self.a1 = Swish()
        self.a2 = Swish()
        self.a3 = Swish()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x - (batchsize, nc, h, w)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3] 
        x = x.permute(0, 2, 3, 1)
        # x - (batchsize, h, w, nc)
        grid = self.get_grid(batchsize, size_x, size_y, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x - (batchsize, width, h, w)
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = self.a0(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = self.a1(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = self.a2(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.a3(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class ConvNetEstimator(nn.Module):
    def __init__(self, in_c, out_c, hidden, factor, net_type):
        super().__init__()

        kernel_size = 3
        padding = kernel_size // 2

        self.out_c = out_c
        self.factor = factor

        if net_type == 'conv':
            bias = True
            self.res = nn.Sequential(
                nn.Conv2d(in_c, hidden, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=bias),
                Swish(),
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=bias),
                Swish(),
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=bias),
                Swish(),
                nn.Conv2d(hidden, out_c, kernel_size=kernel_size, padding=padding, padding_mode='circular'),
            )
        elif net_type == 'fno':
            self.res = FNO2d(6, 6, 10)
        else:
            raise NotImplementedError

    def forward(self, x):
        flatten = x.dim() == 5
        if flatten:
            batch_size, nc, T, h, w = x.shape
            x = x.transpose(2, 1)
            x = x.reshape(batch_size * T, nc, h, w)

        x = self.res(x) * self.factor

        if flatten:
            x = x.reshape(batch_size, T, self.out_c, h, w)
            x = x.transpose(2, 1).contiguous()
        return x