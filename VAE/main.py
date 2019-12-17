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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


args = get_args()
device = "cuda"  # args.device_ids[0]
z_dim = args.z_dim
epochs = args.epochs


def train(epoch):
    time.sleep(0.5)
    for i, x in enumerate(tqdm(train_loader)):
        x = x.to(device)
        loss = model.train({"x": x})
        if (i + 1) % 100 == 0:
            itr = i + train_loader.L * epoch
            writer.add_scalar("loss/itr_train", loss, itr)
        if i + 1 == train_loader.L:
            break
    path = "epoch{:04d}-iter{:05d}-{}".format(epoch, i, loss)
    show(epoch)
    save(path)
    print("Epoch: {} Train loss: {:.4f}".format(epoch, loss))
    writer.add_scalar("loss/train", loss, epoch)


def test(epoch):
    test_loss = 0
    for i, x in enumerate(test_loader):
        x = x.to(device)
        loss = model.test({"x": x})
        test_loss += loss
        if i + 1 == test_loader.L:
            break
    print("Test loss: {:.4f}".format(loss))
    writer.add_scalar("loss/test", loss, epoch)


def plot_reconstrunction(x):
    with torch.no_grad():
        z = q.sample({"x": x}, return_all=False)
        recon_batch = p.sample_mean(z)
        comparison = (
            torch.stack([x[:32], recon_batch[:32]])
            .transpose(0, 1)
            .reshape(64, 3, 64, 64)
        )
        return comparison


def plot_image_from_latent(z_sample):
    with torch.no_grad():
        sample = p.sample_mean({"z": z_sample}).view(-1, 3, 64, 64).cpu()
        return sample


def show(epoch):
    recon = plot_reconstrunction(_x)
    writer.add_images("image/reconst", recon, epoch)
    sample = plot_image_from_latent(z_sample)
    writer.add_images("image/sample", sample, epoch)
    # img = torch.cat([recon[:8], sample[:8]])

    # plt.figure(figsize=(16, 6))
    # for i in range(24):
    #     plt.subplot(3, 8, i + 1)
    #     plt.imshow(
    #         img[i].numpy().astype(np.float).transpose(1, 2, 0).reshape(64, 64, 3)
    #     )
    # plt.savefig("./logs/figure/vae_" + path + ".png")
    # plt.close()


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
model = VAE(q, p, regularizer=kl, optimizer=optim.Adam,
            optimizer_params={"lr": 1e-3})

log_dir = "../runs/" + datetime.now().strftime("%b%d_%H-%M-%S") + "_" + args.comment
writer = SummaryWriter(log_dir=log_dir)

train_loader = PushDataLoader(args.path, "train", args.B, args.epochs)
test_loader = PushDataLoader(args.path, "test", args.B, args.epochs)

z_sample = 0.5 * torch.randn(64, z_dim).to(device)
_x = next(test_loader)
_x = _x.to(device)

del test_loader
test_loader = PushDataLoader(args.path, "test", args.B, args.epochs)

# load(path, device)

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)