# base model for blind SR, input LR, output kernel + SR
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast as autocast
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()

        self.ResBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, padding=3),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.ResBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=5, padding=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.ResBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(in_channels=3*channel, out_channels=channel, kernel_size=3, padding=1) 
        
    def forward(self, x):
        
        residual = x
        x1 = self.ResBlock1(x)
        x1 += residual
        
        x2 = self.ResBlock2(x)
        x2 += residual

        x3 = self.ResBlock3(x)
        x3 += residual

        
        out = torch.cat([x1,x2,x3], axis=1)
        
        out = self.conv(out)

        return out

class SplitModule(nn.Module):
    def __init__(self, channel, split_num):
        super(SplitModule, self).__init__()

        self.channel = channel
        self.split_num = split_num
        self.share = int(self.channel / self.split_num)
        self.mod = int(self.channel % self.split_num)

        self.Match = nn.Conv2d(in_channels=self.channel, out_channels=(self.channel-self.mod), kernel_size=3, padding=1)
        
        self.AffineModule = nn.Sequential(
            nn.Conv2d(in_channels=(int(self.channel/self.split_num)), out_channels=(int(self.channel/self.split_num)), kernel_size=1),
            # nn.BatchNorm2d(int(self.channel/self.split_num)),
            nn.ReLU(),
        )

        self.split_layers = []
        for i in range(self.split_num):
            self.split_layers.append(self.AffineModule)
        # self.split_layers = nn.Sequentail(self.split_layers)

    def forward(self, x):
        x = self.Match(x)

        tmp = 0
        kernel_list = []
        for i in range(self.share, self.channel+self.share, self.share):
            kernel_list.append(x[:,tmp:i,:,:])
            tmp = i

        for j, kernels in enumerate(kernel_list):
            kernel_list[j] = self.split_layers[j](kernels)

        out = torch.cat(kernel_list, axis=1)

        return out

def stable_batch_kernel(batch, l=21, sig=2.6, sig1=2.6, sig2=2.6, theta=0, rate_iso=1.0, scale=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    batch_sigma = np.zeros((batch, 3))
    shifted_l = l - scale + 1
    for i in range(batch):
        batch_kernel[i, :shifted_l, :shifted_l], batch_sigma[i, :] = \
            stable_gaussian_kernel(l=shifted_l, sig=sig, sig1=sig1, sig2=sig2, theta=theta, rate_iso=rate_iso, scale=scale, tensor=False)
    if tensor:
        return torch.FloatTensor(batch_kernel), torch.FloatTensor(batch_sigma)
    else:
        return batch_kernel, batch_sigma

class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig1=2.6, sig2=2.6, theta=0, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scale=3):
        self.l = l
        self.sig = sig
        self.sig1 = sig1
        self.sig2 = sig2
        self.theta = theta
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate_iso = rate_iso
        self.scale = scale

    def __call__(self, random, batch, tensor=False):
        return stable_batch_kernel(batch, l=self.l, sig=self.sig, sig1=self.sig1, sig2=self.sig2, theta=self.theta,
                                       rate_iso=self.rate_iso, scale=self.scale, tensor=tensor)

class KernelEstimation(nn.Module):
    ''' Network of KernelEstimation'''
    def __init__(self, in_nc=3, kernel_size=21, channels=[128, 256, 128, 64, 32], split_num=2):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        self.head = nn.Conv2d(in_channels=in_nc, out_channels=channels[0], kernel_size=3, padding=1, bias=True)
        
        self.RB1 = ResBlock(channel=channels[0])
        self.SP1 = SplitModule(channel=channels[0], split_num=8)
        
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, bias=True)
        self.RB2 = ResBlock(channel=channels[1])
        self.SP2 = SplitModule(channel=channels[1], split_num=8)
        
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[1], kernel_size=3, padding=1, bias=True)
        self.RB3 = ResBlock(channel=channels[2])
        self.SP3 = SplitModule(channel=channels[2], split_num=8)

        # self.tail = nn.Sequential(
        #     nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(channels[3]),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=channels[3], out_channels=441, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(441),
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # self.softmax = nn.Softmax(1)
        self.linear1 = nn.Linear(128, 3)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.PReLU()

    
    # def layer_recalibration(self, layer):

    #     if(torch.isnan(layer.weight).sum() > 0):
    #         print ('recalibrate layer.weight')
    #         layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight)
    #         layer.weight += 1e-7



    def forward(self, x):
        b, c, h, w = x.size()
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        x = self.head(x)
        x = self.RB1(x)
        multiple = self.SP1(x)
        
        # multiple = multiple.permute(0,1,3,2)
        # x = torch.matmul(x, multiple)
        x = x * multiple
        
        x = self.conv1(x)
        x = self.RB2(x)
        multiple2 = self.SP2(x)
        
        # x = torch.matmul(x, multiple2)
        x = x * multiple2
        
        x = self.conv2(x)
        x = self.RB1(x)
        multiple3 = self.SP1(x)
        x = x * multiple3



        # x = torch.matmul(x, multiple3)
        x = self.avg_pool(x)
        x = x.view(b, x.size(1))

        # self.layer_recalibration(self.linear1)
        x = self.linear1(x)
        x = self.relu(x)
        # print('x[0]')
        # print(x[0])
        # m = nn.Threshold(0.7, 10)
        # x = m(x)
        return x