# https://github.com/google-research/planet/blob/master/planet/models/ssm.py
# https://github.com/google-research/planet/blob/master/planet/networks/conv_ha.py

import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Bernoulli, Deterministic


MIN_STDDEV = 1e-5


# --------------------------------
# basic
# --------------------------------

# cond_var=["s_prev", "a_list"] ?


class Prior(Normal):
    def __init__(self, s_dim, a_dim, aa_dim=None):
        if not aa_dim:
            super(Prior, self).__init__(cond_var=["s_prev", "a"], var=["s"])
        else:
            super(Prior, self).__init__(cond_var=["s_prev", "a", "aa"], var=["s"])

        self._min_stddev = MIN_STDDEV
        self.enc_a = nn.Linear(a_dim, s_dim)
        if not aa_dim:
            self.fc1 = nn.Linear(s_dim * 2, s_dim * 2)
        else:
            self.enc_aa = nn.Linear(aa_dim, s_dim)
            self.fc1 = nn.Linear(s_dim * 3, s_dim * 2)
        self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, a, aa=None):
        a = self.enc_a(a)
        if not aa:
            h = torch.cat((s_prev, a), 1)
        else:
            aa = self.enc_aa(aa)
            h = torch.cat((s_prev, a, aa), 1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h),
                "scale": F.softplus(self.fc22(h)) + self._min_stddev}


class Posterior(Normal):
    def __init__(self, prior, h_dim, s_dim, a_dim, aa_dim=None):
        if not aa_dim:
            super(Posterior, self).__init__(cond_var=["s_prev", "a", "h"], var=["s"])
        else:
            super(Posterior, self).__init__(cond_var=["s_prev", "a", "aa", "h"], var=["s"])

        self._min_stddev = MIN_STDDEV
        self.prior = prior
        self.fc1 = nn.Linear(s_dim * 2 + h_dim, s_dim * 2)
        self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, a, aa=None, h=None):
        pri = self.prior(s_prev, a, aa)
        h = torch.cat((pri["loc"], pri["scale"], h), 1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h),
                "scale": F.softplus(self.fc22(h)) + self._min_stddev}


class Encoder(Deterministic):
    def __init__(self):
        super(Encoder, self).__init__(cond_var=["x"], var=["h"])
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
    def __init__(self, s_dim):
        super(Decoder, self).__init__(cond_var=["s"], var=["x"])
        self.fc = nn.Linear(s_dim, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)  # 5x5
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)    # 13x13
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)     # 30x30
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)      # 64x64

    def forward(self, s):
        h = self.fc(s)  # linear
        h = h.view(s.size(0), 1024, 1, 1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.conv4(h)
        return {"loc": h, "scale": 1.0}


# --------------------------------
# proposal
# --------------------------------


class DecoderEnsemble(Normal):
    def __init__(self, s_dim, ss_dim):
        super(DecoderEnsemble, self).__init__(cond_var=["s", "ss"], var=["x"])
        self.fc1 = nn.Linear(s_dim, 1024)
        self.fc2 = nn.Linear(ss_dim, 1024)
        self.conv1 = nn.ConvTranspose2d(2048, 128, 5, stride=2)  # 5x5
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)    # 13x13
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)     # 30x30
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)      # 64x64

    def forward(self, s, ss):
        h1 = self.fc1(s)
        h2 = self.fc2(ss)
        h = torch.cat((h1, h2), 1)
        h = h.view(s.size(0), 2048, 1, 1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.conv4(h)
        return {"loc": h, "scale": 1.0}


class DecoderResidual(Normal):
    def __init__(self, s_dim, ss_dim):
        super(DecoderResidual, self).__init__(cond_var=["s", "ss"], var=["x"])
        self.mode = None
        self.fc1 = nn.Linear(s_dim, 1024)
        self.fc2 = nn.Linear(ss_dim, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)  # 5x5
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)    # 13x13
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)     # 30x30
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)      # 64x64

    def forward(self, s, ss):
        assert self.mode in [0, 1], "not expected"
        h1 = self.fc1(s)
        if self.mode == 1:
            h2 = self.fc2(ss)
            h1 += h2
        self.mode = None
        h = h1.view(s.size(0), 1024, 1, 1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.conv4(h)
        return {"loc": h, "scale": 1.0}