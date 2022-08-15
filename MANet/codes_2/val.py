import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import collections
try:
    import accimage
except ImportError:
    accimage = None

import yaml
from scipy import signal
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import scipy
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from torch.distributions.multivariate_normal import MultivariateNormal



def anisotropic_gaussian_kernel_matlab(l, sig1, sig2, theta, tensor=False):
    # mean = [0, 0]
    # v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    # V = np.array([[v[0], v[1]], [v[1], -v[0]]]) # [[cos, sin], [sin, -cos]]
    # D = np.array([[sig1, 0], [0, sig2]])
    # cov = np.dot(np.dot(V, D), V) # VD(V^-1), V=V^-1
    sig1, sig2, theta = 1,1,1

    cov11 = sig1*np.cos(theta)**2 + sig2*np.sin(theta)**2
    cov22 = sig1*np.sin(theta)**2 + sig2*np.cos(theta)**2
    cov21 = (sig1-sig2)*np.cos(theta)*np.sin(theta)
    cov = np.array([[cov11, cov21], [cov21, cov22]])

    center = 10
    x, y = np.mgrid[-center:-center+l:1, -center:-center+l:1]
    pos = np.dstack((y, x))
    k = scipy.stats.multivariate_normal.pdf(pos, mean=[0, 0], cov=cov)


    k[k < scipy.finfo(float).eps * k.max()] = 0
    sumk = k.sum()
    if sumk != 0:
        k = k/sumk

    return torch.FloatTensor(k) if tensor else k

anisotropic_gaussian_kernel_matlab(21,1,1,0.5)

def anisotropic_gaussian_kernel_tensor(tensor=False):
    # mean = [0, 0]
    # v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    # V = np.array([[v[0], v[1]], [v[1], -v[0]]]) # [[cos, sin], [sin, -cos]]
    # D = np.array([[sig1, 0], [0, sig2]])
    # cov = np.dot(np.dot(V, D), V) # VD(V^-1), V=V^-1
    l = 21
    sig1, sig2, theta = torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 1)
    # sig1, sig2, theta = torch.from_numpy(0.1),torch.from_numpy(0.1),torch.from_numpy(0.5)
    cov11 = sig1*torch.cos(theta)**2 + sig2*torch.sin(theta)**2
    cov22 = sig1*torch.sin(theta)**2 + sig2*torch.cos(theta)**2
    cov21 = (sig1-sig2)*torch.cos(theta)*torch.sin(theta)
    cov = torch.tensor([[cov11, cov21], [cov21, cov22]])

    center = 10
    # x, y = torch.meshgrid[-center:-center+l:1, -center:-center+l:1]
    x, y = np.mgrid[-center:-center+l:1, -center:-center+l:1]
    pos = np.dstack((y, x))
    pos = torch.FloatTensor(pos)
    k = torch.exp(MultivariateNormal(torch.zeros(2), cov).log_prob(pos))

    k[k < torch.finfo(float).eps * k.max()] = 0
    sumk = k.sum()
    if sumk != 0:
        k = k/sumk
    return torch.FloatTensor(k) if tensor else k

anisotropic_gaussian_kernel_tensor()