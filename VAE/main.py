from pixyz.losses import KullbackLeibler
from pixyz.models import VAE
from pixyz.distributions import Normal
import torch
from torch import nn, optim
from data_loader import PushDataLoader
from core import Inference, Generator
from config import get_args
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


args = get_args()
device = "cuda"  # args.device_ids[0]
z_dim = args.z_dim
epochs = args.epochs


def train(epoch):
    time.sleep(0.5)
    flag_endepoch = False
    for i, x in enumerate(tqdm(train_loader)):
        if flag_endepoch:
            break
        x = x.to(device)
        for t in range(args.T):
            loss = model.train({"x": x[:, t]})
        if i % 100 == 0:
            print("loss: {:.4f}".format(loss))
            path = "epoch{:04d}-iter{:05d}-{}".format(epoch, i, loss)
            show(path)
            save(path)
        if i + 1 == train_loader.length:
            flag_endepoch = True
    print("Epoch: {} Train loss: {:.4f}".format(epoch, loss))


def test(epoch):
    test_loss = 0
    flag_endepoch = False
    for i, x in enumerate(test_loader):
        if flag_endepoch:
            break
        x = x.to(device)
        for t in range(args.T):
            loss = model.test({"x": x[:, t]})
            test_loss += loss
        if i + 1 == test_loader.length:
            flag_endepoch = True
    print("Test loss: {:.4f}".format(loss))


def plot_reconstrunction(x):
    with torch.no_grad():
        z = q.sample({"x": x}, return_all=False)
        recon_batch = p.sample_mean(z).view(-1, 3, 64, 64)
        comparison = torch.cat([x.view(-1, 3, 64, 64), recon_batch]).cpu()
        return comparison


def plot_image_from_latent(z_sample):
    with torch.no_grad():
        sample = p.sample_mean({"z": z_sample}).view(-1, 3, 64, 64).cpu()
        return sample


def show(path):
    recon = plot_reconstrunction(_x[:8])
    sample = plot_image_from_latent(z_sample)
    img = torch.cat([recon, sample[:8]])

    plt.figure(figsize=(16, 6))
    for i in range(24):
        plt.subplot(3, 8, i + 1)
        plt.imshow(
            img[i].numpy().astype(np.float).transpose(1, 2, 0).reshape(64, 64, 3)
        )
    plt.savefig("./logs/figure/vae_" + path + ".png")
    # plt.show()


def save(path):
    torch.save(p.state_dict(), "./logs/p/" + path + ".pt")
    torch.save(q.state_dict(), "./logs/q/" + path + ".pt")
    torch.save(model.optimizer.state_dict(), "./logs/opt/" + path + ".pt")


def load(path, device):
    if not path[-3:] == ".pt":
        path += ".pt"
    p.load_state_dict(torch.load("./logs/p/" + path, map_location=device))
    q.load_state_dict(torch.load("./logs/q/" + path, map_location=device))
    model.optimizer.load_state_dict(
        torch.load("./logs/opt/" + path, map_location=device)
    )


p = Generator(z_dim).to(device)
q = Inference(z_dim).to(device)

prior = Normal(
    loc=torch.tensor(0.0),
    scale=torch.tensor(1.0),
    var=["z"],
    features_shape=[z_dim],
    name="p_{prior}",
).to(device)
kl = KullbackLeibler(q, prior)
model = VAE(q, p, regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})

train_loader = PushDataLoader(args.path, "train", args.B, args.epochs)
test_loader = PushDataLoader(args.path, "test", args.B, args.epochs)

z_sample = 0.5 * torch.randn(64, z_dim).to(device)
_x = next(test_loader)[:, 0]
_x = _x.to(device)

# load(path, device)

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)