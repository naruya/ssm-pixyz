# https://github.com/akanimax/Variational_Discriminator_Bottleneck/blob/master/source/vdb/Gan_networks.py

import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Deterministic
import numpy as np


DIM = 128


class Prior(Normal):
    def __init__(self, s_dim, a_dims, min_stddev=0.):
        super(Prior, self).__init__(cond_var=["s_prev", "a_list"], var=["s"])
        self.min_stddev = min_stddev

        self.fc_loc11 = nn.Linear(s_dim + sum(a_dims), DIM)
        self.fc_loc12 = nn.Linear(DIM, s_dim)

        self.fc_loc21 = nn.Linear(s_dim * 2 + sum(a_dims), DIM)
        self.fc_loc22 = nn.Linear(DIM, s_dim)

        self.fc_loc31 = nn.Linear(s_dim * 2 + sum(a_dims), DIM)
        self.fc_loc32 = nn.Linear(DIM, s_dim)

        self.fc_loc41 = nn.Linear(s_dim * 2 + sum(a_dims), DIM)
        self.fc_loc42 = nn.Linear(DIM, s_dim)

        self.fc_scale11 = nn.Linear(DIM * 4, DIM)
        self.fc_scale12 = nn.Linear(DIM, s_dim)

    def forward_shared(self, s_prev, a_list):
        # 全部 s_t = s_t + foo(s_prev, a) だと、長期ステップで死ぬ
        # 全部 s_t = foo(s_prev, a) だと、でかいs_dimで死ぬ
        h = torch.cat([s_prev] + a_list, 1)
        h1 = F.relu(self.fc_loc11(h))
        s1 = self.fc_loc12(h1)

        h = torch.cat([s1, s1 - s_prev] + a_list, 1)
        h2 = F.relu(self.fc_loc21(h))
        s2 = s1 + 0.1 * self.fc_loc22(h2)

        h = torch.cat([s2, s2 - s1] + a_list, 1)
        h3 = F.relu(self.fc_loc31(h))
        s3 = s2 + 0.1 * self.fc_loc32(h3)

        h = torch.cat([s3, s3 - s2] + a_list, 1)
        h4 = F.relu(self.fc_loc41(h))
        s4 = s3 + 0.1 * self.fc_loc42(h4)

        loc = s4

        h = torch.cat([h1, h2, h3, h4], 1)
        scale = self.fc_scale12(F.relu(self.fc_scale11(h)))

        return loc, scale

    def forward(self, s_prev, a_list):
        loc, scale = self.forward_shared(s_prev, a_list)
        return {"loc": loc,
                "scale": F.softplus(scale) + self.min_stddev}


class Posterior(Normal):
    def __init__(self, prior, s_dim, h_dim, a_dims, min_stddev=0.):
        super(Posterior, self).__init__(cond_var=["s_prev", "h", "a_list"], var=["s"])
        self.min_stddev = min_stddev
        self.prior = prior
        self.fc1 = nn.Linear(s_dim * 2 + h_dim, DIM)
        self.fc21 = nn.Linear(DIM, s_dim)
        self.fc22 = nn.Linear(DIM, s_dim)

    def forward(self, s_prev, h, a_list):
        loc, scale = self.prior.forward_shared(s_prev, a_list)
        h = torch.cat([loc, scale, h], 1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h),
                "scale": F.softplus(self.fc22(h)) + self.min_stddev}


class Encoder(Deterministic):
    def __init__(self, size=64, num_filters=64, max_filters=256):
        super(Encoder, self).__init__(cond_var=["x"], var=["h"])

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


class Decoder(Normal):
    def __init__(self, s_dim, size=64, final_channels=64, max_channels=256):
        super(Decoder, self).__init__(cond_var=["s"], var=["x"])

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

        return {"loc": out, "scale": out.new_ones(out.size())}


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
    # out = F.leaky_relu(x, 2e-1)
    out = F.relu(x)
    return out