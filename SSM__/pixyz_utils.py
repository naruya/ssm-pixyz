import os
import torch
from datetime import datetime


def save_model(model):
    name = model.__class__.__name__
    time = datetime.now().strftime("%b%d_%H-%M-%S")
    path = os.path.join("model", name + "_" + comment, time)
    os.makedirs(path, exist_ok=True)
    for i, dist in enumerate(model.distributions):
        torch.save(dist.state_dict(), os.path.join(path, "dist" + str(i) + ".pt"))
    torch.save(model.optimizer.state_dict(), os.path.join(path, "opt.pt"))


def load_model(model, time):
    name = model.__class__.__name__
    path = os.path.join("model", name, time)
    for i, dist in enumerate(model.distributions):
        dist.load_state_dict(torch.load(os.path.join(path, "dist" + str(i) + ".pt")))
    model.optimizer.load_state_dict(torch.load(os.path.join(path, "opt.pt")))