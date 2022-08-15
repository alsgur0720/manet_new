import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
import numpy as np

class transformEstimator(nn.Module):
    def __init__(self):
        super(transformEstimator, self).__init__()
        self.conv1_k1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_k1 = nn.BatchNorm2d(16)
        self.conv2_k1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_k1 = nn.BatchNorm2d(32)
        self.conv3_k1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_k1 = nn.BatchNorm2d(64)
        
        self.conv1_k2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_k2 = nn.BatchNorm2d(16)
        self.conv2_k2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_k2 = nn.BatchNorm2d(32)
        self.conv3_k2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_k2 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(16)
        
        self.linear1 = nn.Linear(16, 8)
        self.linear2 = nn.Linear(8, 6)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # self.out = nn.Conv2d(in_channels=33, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
    
    def forward(self, x1, x2):
        B,_,_,_ = x1.size()
        temp1 = x1.clone()
        temp2 = x2.clone()
        
        k1 = self.conv1_k1(self.relu(x1))
        k2 = self.conv1_k2(self.relu(x2))
        
        k2 += k1
        
        k1 = self.conv2_k1(self.relu(k1))
        k2 = self.conv2_k2(self.relu(k2))
        
        k2 += k1
        
        k1 = self.conv3_k1(self.relu(k1))
        k2 = self.conv3_k2(self.relu(k2))
        
        k2 = k1 + k2 # torch.Size([32, 64, 6, 6])
        
        k2 = self.conv4(k2)
        k2 = self.conv5(k2)
        out = self.avg_pool(k2)
        out = out.view([B,out.size()[1]])
        out = self.linear1(out)
        out = self.linear2(out)
        # out = self.sigmoid(out)
        
        return out

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        print('\nMaking model...')
        self.opt = opt
        self.n_GPUs = opt.n_GPUs
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.model = transformEstimator().to(self.device)
        self.load(opt.pre_train, cpu=opt.cpu)

    def load(self, pre_train, cpu=False):
        
        #### load gaze model ####
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.model.load_state_dict(
                torch.load(pre_train),
                strict=True
            )
            print("Complete loading model weight")
        
        num_parameter = self.count_parameters(self.model)

        print(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M \n")

    def forward(self, x1, x2):
        return self.model(x1, x2)
    
    def count_parameters(self, model):
        param_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return param_sum