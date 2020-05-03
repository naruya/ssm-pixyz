# https://github.com/akanimax/Variational_Discriminator_Bottleneck/blob/master/source/vdb/Gan_networks.py

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from logzero import logger


# Normal
class ResPrior(nn.Module):
    def __init__(self, k, s_dim, a_dims, min_stddev=0.):
        super(ResPrior, self).__init__()
        self.name = str(k) + "-" + self.__class__.__name__.lower()
        self.min_stddev = min_stddev
        self.fc1 = nn.Linear(s_dim + sum(a_dims), s_dim * 2)
        # self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc_res1 = nn.Linear(s_dim * 2, s_dim)
        # self.fc_res2 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, a_list):
        h = torch.cat([s_prev] + a_list, 1)
        h = F.relu(self.fc1(h))
        # h1 = self.fc21(h)
        dx = self.fc_res1(actvn(h))
        # dx = self.fc_res2(actvn(dx))
        h1 = s_prev + 0.1 * dx
        h2 = self.fc22(h)
        logger.debug("p fc21: {:12.6f} ".format(torch.max(torch.abs(h1.data)).item()))
        logger.debug("p fc22: {:12.6f} ".format(torch.max(torch.abs(h2.data)).item()))
        loc = h1
        scale = F.softplus(h2) + self.min_stddev
        return loc, scale


# Normal
# posteriorはresじゃない方がいいかも？
class ResPosterior(nn.Module):
    def __init__(self, k, s_dim, h_dim, a_dims, min_stddev=0.):
        super(ResPosterior, self).__init__()
        self.name = str(k) + "-" + self.__class__.__name__.lower()
        self.min_stddev = min_stddev
        self.fc1 = nn.Linear(s_dim + h_dim + sum(a_dims), s_dim * 2)
        # self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc_res1 = nn.Linear(s_dim * 2, s_dim)
        # self.fc_res2 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, h_t, a_list):
        h = torch.cat([s_prev, h_t] + a_list, 1)
        h = F.relu(self.fc1(h))
        # h1 = self.fc21(h)
        dx = self.fc_res1(actvn(h))
        # dx = self.fc_res2(actvn(dx))
        h1 = s_prev + 0.1 * dx
        h2 = self.fc22(h)
        logger.debug("q fc21: {:12.6f} ".format(torch.max(torch.abs(h1.data)).item()))
        logger.debug("q fc22: {:12.6f} ".format(torch.max(torch.abs(h2.data)).item()))
        loc = h1
        scale = F.softplus(h2) + self.min_stddev
        return loc, scale


def actvn(x):
    """
    utility helper for leaky Relu activation
    :param x: input tensor
    :return: activation applied tensor
    """
    out = F.leaky_relu(x, 2e-1)
    return out