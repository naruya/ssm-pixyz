import os
import torch
from datetime import datetime


def save_model(model):
    time = datetime.now().strftime("%b%d_%H-%M-%S")
    path = os.path.join("model", model.args.timestamp, time)
    os.makedirs(path, exist_ok=True)
    for i, dist in enumerate(model.distributions):
        torch.save(dist.state_dict(), os.path.join(path, "dist" + str(i) + ".pt"))
    torch.save(model.discriminator.state_dict(), os.path.join(path, "discriminator.pt"))
    torch.save(model.g_optimizer.state_dict(), os.path.join(path, "g_opt.pt"))
    torch.save(model.d_optimizer.state_dict(), os.path.join(path, "d_opt.pt"))


def load_model(model, time):
    path = os.path.join("model", model.args.timestamp, time)
    for i, dist in enumerate(model.distributions):
        dist.load_state_dict(torch.load(os.path.join(path, "dist" + str(i) + ".pt")))
    model.g_optimizer.load_state_dict(torch.load(os.path.join(path, "g_opt.pt")))
    model.d_optimizer.load_state_dict(torch.load(os.path.join(path, "d_opt.pt")))
    model.discriminator.load_state_dict(torch.load(os.path.join(path, "discriminator.pt")))