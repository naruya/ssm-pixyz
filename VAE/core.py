import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Bernoulli
# from torch_utils import Conv2dLSTM

# https://github.com/iShohei220/corl-gqn/blob/taniguchi/vae/vae.py

# inference model q(z|x)
class Inference(Normal):
    def __init__(self, z_dim):
        super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0), # (N,3,64,64) -> (N,64,60,60)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (N,64,60,60) -> (N,64,30,30)
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=0), # (N,64,30,30) -> (N,32,26,26)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (N,32,26,26) -> (N,32,13,13)
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), # (N,32,13,13) -> (N,32,9,9)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (N,32,9,9) -> (N,32,4,4)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, 512),  # input size should be (N,3,64,64)
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(512, z_dim)
        self.fc2 = nn.Linear(512, z_dim)
        _init_weights(self)

    def forward(self, x):
        h = self.fc(self.conv(x).view(-1, 32*4*4))
        return {"loc": self.fc1(h), "scale": F.softplus(self.fc2(h))}


# generative model p(x|z)
class Generator(Bernoulli):
    def __init__(self, z_dim):
        super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32*5*5),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=0), # (N,32,5,5) -> (N,32,13,13)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=0, output_padding=1), # (N,32,13,13) -> (N,64,30,30)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=0, output_padding=1), # (N,64,30,30) -> # (N,3,64,64)
        )
        _init_weights(self)

    def forward(self, z):
        h = self.conv(self.fc(z).view(-1, 32, 5, 5))
        return {"probs": torch.sigmoid(h)}
    

def _init_weights(distribution):
#     print("init weights")
    for m in distribution.modules():
#         print(type(m), end=" ")
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.normal_(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # debug
#         else:
#             print("not initialized")
#             continue
#         print("ok")