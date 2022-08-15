import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils.util import anisotropic_gaussian_kernel_matlab
import sys

class Cus_Loss(nn.modules.loss._Loss):
    def __init__(self, opt):
        super(Cus_Loss, self).__init__()
        self.opt = opt
        self.loss = []
        self.loss_module = nn.ModuleList()
        self.L1_Loss = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss()

    def forward(self, params, gt_param, real_k):
        b = params.size(0)
        # print(params[0:3])
        # print(gt_param[0:3])
        # sys.exit()
        fake_K = torch.zeros((b, self.opt['kernel_size'], self.opt['kernel_size'])).to('cuda:0')
        shifted_l = self.opt['kernel_size'] - self.opt['scale'] + 1
        for i in range(b):
            # print(params[i].detach().cpu().numpy())
            # sys.exit()
            sig1, sig2, theta = params[i].detach().cpu().numpy()
            sig1 += 0.7
            sig2 += 0.7
            fake_K[i, :shifted_l, :shifted_l] = \
                anisotropic_gaussian_kernel_matlab( \
                    l=shifted_l, \
                    sig1=sig1, \
                    sig2=sig2, \
                    theta=theta, tensor=True)
        # self.fake_K = self.real_K
        kenel_loss = self.L1_Loss(fake_K * 1000, real_k * 1000)
        
        params_loss = self.mse(params, gt_param)

        return params_loss, fake_K
