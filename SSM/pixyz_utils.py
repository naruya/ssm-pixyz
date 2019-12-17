import os
import torch
from torch_utils import init_weights


def save_model(model, prefix):
    name = model.__class__.__name__
    path = os.path.join("model", name, prefix)
    os.makedirs(path, exist_ok=True)
    for i, dist in enumerate(model.distributions):
        torch.save(dist.state_dict(), os.path.join(path, "dist" + str(i) + ".pt"))
    torch.save(model.optimizer.state_dict(), os.path.join(path, "opt.pt"))


def load_model(model, prefix):
    name = model.__class__.__name__
    path = os.path.join("model", name, prefix)
    for i, dist in enumerate(model.distributions):
        dist.load_state_dict(torch.load(os.path.join(path, "dist" + str(i) + ".pt")))
    model.optimizer.load_state_dict(torch.load(os.path.join(path, "opt.pt")))


def eval_mode(model):
    for dist in model.distributions:
        dist.eval()


def train_mode(model):
    for dist in model.distributions:
        dist.train()


# def init_model(model):
#     for dist in model.distributions:
#         init_weights(dist)