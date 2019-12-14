import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Bernoulli, Deterministic

# from torch_utils import Conv2dLSTM

# --------------------------------
# SSM (base)
# --------------------------------


class Prior_S(Normal):
    def __init__(self, s_dim, a_dim):
        super(Prior_S, self).__init__(cond_var=["s_prev", "a"], var=["s"])
        self.upsample_a = nn.Linear(a_dim, s_dim)
        self.fc1 = nn.Linear(s_dim * 2, s_dim * 2)
        self.fc21 = nn.Linear(s_dim * 2, s_dim)
        self.fc22 = nn.Linear(s_dim * 2, s_dim)

    def forward(self, s_prev, a):
        a = self.upsample_a(a)
        h = torch.cat((s_prev, a), 1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


class Decoder_S(Bernoulli):
    def __init__(self, s_dim):
        super(Decoder_S, self).__init__(cond_var=["s"], var=["x"])
        self.fc = nn.Sequential(
            nn.Linear(s_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 5 * 5),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 5, stride=2, padding=0),  # -> (N,32,13,13)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 64, 5, stride=2, padding=0, output_padding=1
            ),  # -> (N,64,30,30)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 3, 5, stride=2, padding=0, output_padding=1
            ),  # -> # (N,3,64,64)
        )

    def forward(self, s):
        h = self.conv(self.fc(s).view(-1, 32, 5, 5))
        return {"probs": torch.sigmoid(h)}


class EncoderRNN_S(Deterministic):
    def __init__(self, h_dim, device):
        super(EncoderRNN_S, self).__init__(cond_var=["x"], var=["h"])
        self.h_dim = h_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=0),  # -> (N,64,60,60)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),  # -> (N,64,30,30)
            nn.Conv2d(64, 32, 5, stride=1, padding=0),  # -> (N,32,26,26)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),  # -> (N,32,13,13)
            nn.Conv2d(32, 32, 5, stride=1, padding=0),  # -> (N,32,9,9)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),  # -> (N,32,4,4)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 512),  # input size should be (N,3,64,64)
            nn.ReLU(),
            nn.Linear(512, h_dim),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(h_dim, h_dim)
        self.h0 = nn.Parameter(torch.zeros(1, 1, h_dim)).to(device)
        self.c0 = nn.Parameter(torch.zeros(1, 1, h_dim)).to(device)

    def forward(self, x):
        _T = x.size(0)
        _B = x.size(1)
        h0 = self.h0.expand(1, _B, self.h_dim).contiguous()
        c0 = self.c0.expand(1, _B, self.h_dim).contiguous()
        x = x.reshape(-1, 3, 64, 64)  # TxBxCxHxW => T*BxCxHxW
        h = self.fc(self.conv(x).view(-1, 32 * 4 * 4))
        h = h.reshape(_T, -1, h.size(-1))  # T*Bx32 => TxBx32
        h, _ = self.rnn(h, (h0, c0))  # TxBx32, (1xBx32, 1xBx32)
        return {"h": h}


class Inference_S(Normal):
    def __init__(self, h_dim, s_dim, a_dim):
        super(Inference_S, self).__init__(cond_var=["h", "s_prev", "a"], var=["s"])
        self.upsample_a = nn.Linear(a_dim, s_dim)
        self.fc3 = nn.Linear(h_dim + s_dim * 2, h_dim + s_dim * 2)
        self.fc41 = nn.Linear(h_dim + s_dim * 2, s_dim)
        self.fc42 = nn.Linear(h_dim + s_dim * 2, s_dim)

    def forward(self, h, s_prev, a):
        a = self.upsample_a(a)
        h = torch.cat((h, s_prev, a), 1)
        h = F.relu(self.fc3(h))
        return {"loc": self.fc41(h), "scale": F.softplus(self.fc42(h))}