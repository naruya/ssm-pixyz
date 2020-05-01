# https://github.com/akanimax/Variational_Discriminator_Bottleneck/blob/master/source/vdb/Gan_networks.py

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Deterministic


class ResnetBlock(nn.Module):
    """
    Resnet Block Sub-module for the Generator and the Discriminator
    Args:
        :param fin: number of input filters
        :param fout: number of output filters
        :param fhidden: number of filters in the hidden layer
        :param is_bias: whether to use affine conv transforms
    """

    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        """ derived constructor """

        # call to super constructor
        super().__init__()

        # State of the object
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout

        # derive fhidden if not given
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Subsubmodules required by this submodule
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x, alpha=0.1):
        """
        forward pass of the block
        :param x: input tensor
        :param alpha: weight of the straight path
        :return: out => output tensor
        """
        # calculate the shortcut path
        x_s = self._shortcut(x)

        # calculate the straight path
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))

        # combine the two paths via addition
        out = x_s + alpha * dx  # note the use of alpha weighter

        return out

    def _shortcut(self, x):
        """
        helper to calculate the shortcut (residual) computations
        :param x: input tensor
        :return: x_s => output tensor from shortcut path
        """
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    """
    utility helper for leaky Relu activation
    :param x: input tensor
    :return: activation applied tensor
    """
    out = F.leaky_relu(x, 2e-1)
    return out


class ResEncoder(Deterministic):
    def __init__(self, k, size=64, num_filters=64, max_filters=1024):
        super(ResEncoder, self).__init__(cond_var=["x"], var=["h"])
        self.name = str(k) + "-" + self.__class__.__name__.lower()
        
        # make sure that the max_filters are divisible by 2
        assert max_filters % 2 == 0, "Maximum filters is not an even number"
        assert num_filters % 2 == 0, "Num filters in first layer is not an even number"
        assert size >= 4, "No point in generating images smaller than (4 x 4)"
        assert size & (size - 1) == 0, "size is not a power of 2"

        # state of the object and shorthands
        s0 = self.s0 = 4
        nf = self.nf = num_filters
        nf_max = self.nf_max = max_filters

        # Submodules required by this module
        num_layers = int(np.log2(size / s0))

        # create the block for the Resnet
        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(num_layers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        # resnet module
        self.resnet = nn.Sequential(*blocks)

        # initial image to volume converter
        self.conv_img = nn.Conv2d(3, nf, 3, padding=1)

        # conv_converter for information bottleneck
        nf1 = blocks[-1].conv_1.out_channels  # obtain the final number of channels
        self.conv_converter = nn.Conv2d(nf1, nf1, kernel_size=4, padding=0)

        # final predictions maker
        self.fc = nn.Linear(nf1, 1024)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input image tensor [Batch_size x 3 x height x width]
        :param mean_mode: decides whether to sample points or use means directly
        :return: prediction scores (Linear), mus and sigmas: [Batch_size x 1]
        """

        # convert image to initial volume
        out = self.conv_img(x)

        # apply the resnet module
        out = self.resnet(out)

        # apply the converter
        out = self.conv_converter(actvn(out))

        # flatten the volume
        out = out.squeeze(-1).squeeze(-1)

        # apply the final fully connected layer
        out = self.fc(actvn(out))

        return {"h": out}


class ResDecoder(Normal):
    def __init__(self, k, s_dim, size=64, final_channels=64, max_channels=1024):
        super(ResDecoder, self).__init__(cond_var=["s"], var=["x"])
        self.name = str(k) + "-" + self.__class__.__name__.lower()

        # some peculiar assertions
        assert size >= 4, "No point in generating images less than (4 x 4)"
        assert size & (size - 1) == 0, "size is not a power to 2"

        # state of the object (with some shorthands)
        s0 = self.s0 = 4
        nf = self.nf = final_channels
        nf_max = self.nf_max = max_channels
        self.s_dim = s_dim

        # Submodules required by this module
        num_layers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** num_layers)

        self.fc = nn.Linear(s_dim, self.nf0 * s0 * s0)

        # create the Residual Blocks
        blocks = []  # initialize to empty list
        for i in range(num_layers):
            nf0 = min(nf * 2 ** (num_layers - i), nf_max)
            nf1 = min(nf * 2 ** (num_layers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)

        # final volume to image converter
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
        
        
    def forward(self, s):
        """
        forward pass of the network
        :param z: input z (latent vector) => [Batchsize x latent_size]
        :return:
        """

        batch_size = s.size(0)

        # first layer (Fully Connected)
        out = self.fc(s)
        # Reshape output into volume
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        # apply the Resnet architecture
        out = self.resnet(out)

        # apply the final image converter
        out = self.conv_img(actvn(out))
        out = torch.tanh(out)  # our pixel values are in range [-1, 1]

        return {"loc": out, "scale": 1.0}