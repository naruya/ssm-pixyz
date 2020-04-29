import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Deterministic
from logzero import logger


class Prior(Normal):
    def __init__(self, k, s_dim, a_dims, min_stddev=0.):
        super(Prior, self).__init__(cond_var=["s_prev", "a_list"], var=["s"])
        self.name = str(k) + "-" + self.__class__.__name__.lower()
        self.min_stddev = min_stddev
        self.fc1 = nn.Linear(s_dim + sum(a_dims), s_dim * 2)
        self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, a_list):
        h = torch.cat([s_prev] + a_list, 1)
        h = F.relu(self.fc1(h))
        h1 = self.fc21(h)
        h2 = self.fc22(h)
        logger.debug("prior fc21: {:12.6f} ".format(torch.max(torch.abs(h1.data)).item()))
        logger.debug("prior fc22: {:12.6f} ".format(torch.max(torch.abs(h2.data)).item()))
        return {"loc": h1,
                "scale": F.softplus(h2) + self.min_stddev}


class Posterior(Normal):
    def __init__(self, k, s_dim, h_dim, a_dims, min_stddev=0.):
        super(Posterior, self).__init__(cond_var=["s_prev", "h", "a_list"], var=["s"])
        self.name = str(k) + "-" + self.__class__.__name__.lower()
        self.min_stddev = min_stddev
        self.fc1 = nn.Linear(s_dim + h_dim + sum(a_dims), s_dim * 2)
        self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, h, a_list):
        h = torch.cat([s_prev, h] + a_list, 1)
        h = F.relu(self.fc1(h))
        h1 = self.fc21(h)
        h2 = self.fc22(h)
        logger.debug("posterior fc21: {:12.6f} ".format(torch.max(torch.abs(h1.data)).item()))
        logger.debug("posterior fc22: {:12.6f} ".format(torch.max(torch.abs(h2.data)).item()))
        return {"loc": h1,
                "scale": F.softplus(h2) + self.min_stddev}


class Encoder(Deterministic):
    def __init__(self, k):
        super(Encoder, self).__init__(cond_var=["x"], var=["h"])
        self.name = str(k) + "-" + self.__class__.__name__.lower()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

    def forward(self, x):
        h = F.relu(self.conv1(x))  # 31x31
        h = F.relu(self.conv2(h))  # 14x14
        h = F.relu(self.conv3(h))  # 6x6
        h = F.relu(self.conv4(h))  # 2x2
        h = h.view(x.size(0), 1024)
        return {"h": h}


class Decoder(Normal):
    def __init__(self, k, s_dim):
        super(Decoder, self).__init__(cond_var=["s"], var=["x"])
        self.name = str(k) + "-" + self.__class__.__name__.lower()
        self.fc = nn.Linear(s_dim, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, s):
        h = self.fc(s)  # linear
        h = h.view(s.size(0), 1024, 1, 1)
        h = F.relu(self.conv1(h))  # 5x5
        h = F.relu(self.conv2(h))  # 13x13
        h = F.relu(self.conv3(h))  # 30x30
        h = self.conv4(h)          # 64x64
        return {"loc": h, "scale": 1.0}