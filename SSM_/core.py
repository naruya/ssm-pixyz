# https://github.com/google-research/planet/blob/master/planet/models/ssm.py
# https://github.com/google-research/planet/blob/master/planet/networks/conv_ha.py

import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Bernoulli, Deterministic

# --------------------------------
# SSM
# --------------------------------


class Prior(Normal):
    def __init__(self, s_dim, a_dim):
        super(Prior, self).__init__(cond_var=["s_prev", "a"], var=["s"])
        self.enc_a = nn.Linear(a_dim, s_dim)
        self.fc1 = nn.Linear(s_dim * 2, s_dim * 2)
        self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, a):
        a = self.enc_a(a)
        h = torch.cat((s_prev, a), 1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


class Posterior(Normal):
    def __init__(self, s_dim, a_dim, h_dim):
        super(Posterior, self).__init__(cond_var=["s_prev", "a", "h"], var=["s"])
        self.enc_a = nn.Linear(a_dim, s_dim)
        self.fc1 = nn.Linear(s_dim * 2 + h_dim, s_dim * 2)
        self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, a, h):
        a = self.enc_a(a)
        h = torch.cat((s_prev, a, h), 1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


class Encoder(Deterministic):
    def __init__(self):
        super(Encoder, self).__init__(cond_var=["x"], var=["h"])
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
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
# SSM7~
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
        h1 = F.relu(self.fc1(s))
        h2 = F.relu(self.fc2(ss))
        h = torch.cat((h1, h2), 1)
        h = h.view(s.size(0), 2048, 1, 1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.conv4(h)
        return {"loc": h, "scale": 1.0}