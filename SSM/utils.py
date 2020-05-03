import os
import torch
from torch import nn
import numpy as np
import requests
import json
from logzero import logger
from config import *


# data_parallel ver (dist.module.name)

def save_model(model, save_dir, epoch):
    path = os.path.join(save_dir, "epoch-{:05}".format(epoch))
    os.makedirs(path, exist_ok=True)
    for i, dist in enumerate(model.distributions):
        torch.save(dist.state_dict(), os.path.join(path, dist.module.name + ".pt"))
    torch.save(model.optimizer.state_dict(), os.path.join(path, "optim.pt"))


def load_model(model, load_dir, epoch, load_optim=True):
    logger.debug("---- load model ----")
    path = os.path.join(load_dir, "epoch-{:05}".format(epoch))
    files = os.listdir(path)
    for i, dist in enumerate(model.all_distributions):
        file = dist.module.name + ".pt"
        if file in files:
            dist.load_state_dict(torch.load(os.path.join(path, file)))
            logger.debug("{} found!".format(file))
        else:
            logger.debug("{} NOT found".format(file))
    if load_optim:
        file = "optim.pt"
        logger.debug("---- load optim ----")
        model.optimizer.load_state_dict(torch.load(os.path.join(path, file)))
        logger.debug("{} found!".format(file))


# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
def init_weights(model):
    logger.debug("---- init weights ----")
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.RNN, nn.RNNCell, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            logger.debug("   " + str(type(m)))
            continue
        logger.debug("ok " + str(type(m)))


def check_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug(str(name.ljust(40)) + \
                  "param: {:12.6f} ".format(torch.max(torch.abs(param.data)).item()) + \
                  "grad: {:12.6f} ".format(torch.max(torch.abs(param.grad)).item()))


def flatten_dict(inp_dict):
    return_dict = {}
    for k, v in inp_dict.items():
        if type(v) is list:
            return_dict.update({k + "_" + str(i) : v[i].item() for i in range(len(v))})
        else:
            return_dict.update({k : v.item()})
    return return_dict


def make_summ(info):
    keys = info.keys()
    summ = dict(zip(keys, [0.] * len(keys)))
    return summ


def update_summ(summ, info, _B):
    if summ is None:
        summ = make_summ(info)
    for k in summ.keys():
        summ[k] += info[k] * _B
    return summ


def mean_summ(summ, N):
    for k, v in summ.items():
        summ[k] = v / float(N)
    return summ


def rename_summ(summ, prefix="", suffix=""):
    keys, values = summ.keys(), summ.values()
    keys = prefix + np.array(list(keys), dtype=object) + suffix
    summ = dict(zip(keys, values))
    return summ


def slack(text):
    # webhook_url: "https://hooks.slack.com/services/foo/bar/foobar"
    with open(".slack.txt") as f:
        webhook_url = f.read()
    requests.post(webhook_url, data = json.dumps({"text": text}))