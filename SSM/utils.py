import os
import torch
from torch import nn
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


def flatten_dict(info):
    return_dict = {}
    for k, v in info.items():
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


def write_summ(summ, video, writer, N, epoch, train):
    prefix = "train_" if train else "test_"
    for k, v in summ.items():
        writer.add_scalar("epoch/" + prefix + k, v, epoch)
    if video:
        writer.add_video("epoch/" + prefix + "video", video, epoch)


def slack(text):
    # webhook_url: "https://hooks.slack.com/services/foo/bar/foobar"
    with open(".slack.txt") as f:
        webhook_url = f.read()
    requests.post(webhook_url, data = json.dumps({"text": text}))


# # load all weigts
# def doredakke():
#     args = get_args()
#     logzero.loglevel(args.loglevel)
#
#     TRAIN_INTERVAL = int(43264 / args.B)
#     TEST_INTERVAL = int(256 / args.B)
#     device = args.device_ids[0]
#
#     SEED = args.seed
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed(SEED)
#     torch.backends.cudnn.deterministic = True
#
#     torch.autograd.set_detect_anomaly(True)
#
#     print(args.log_dir)
#
#     writer = None
#
#     paths = os.listdir("_model/Feb18_23-43-15_SSM_s1024_88a3785")
#
#     epoch=1
#     for path in paths:
#         print(path)
#         model = SSM(args, device)
#         path = os.path.join("_model", "Feb18_23-43-15_SSM_s1024_88a3785", path)
#         for i, dist in enumerate(model.distributions):
#             dist.load_state_dict(torch.load(os.path.join(path, "dist" + str(i) + ".pt")))
#         model.optimizer.load_state_dict(torch.load(os.path.join(path, "opt.pt")))
#
#         test_loader = PushDataLoader(args, split="test", shuffle=False)
#         data_loop(epoch, test_loader, model, args.T, device, writer, train=False)


# # resume
#     if args.resume:
#         load_model(model, args.resume_name, args.resume_time)
#
#     path = os.path.join("_model", "Feb18_23-43-15_SSM_s1024_88a3785", "Feb19_03-00-40")
#     for i, dist in enumerate(model.distributions):
#         dist.load_state_dict(torch.load(os.path.join(path, "dist" + str(i) + ".pt")))
#     model.optimizer.load_state_dict(torch.load(os.path.join(path, "opt.pt")))

#     s_dim = "s" + str(args.s_dim[0])
#     if len(args.s_dim) > 1:
#         for i, d in enumerate(args.s_dim[1:]):
#             s_dim += "-" + str(d)
#
#     if args.resume:
#         assert args.resume_name and args.resume_time and args.resume_itr and args.resume_epoch, "invalid resume options"
#
#     if not args.resume:
#         log_dir = os.path.join(
#             args.runs_dir,
#             datetime.now().strftime("%b%d_%H-%M-%S")
#             + "_" + args.model + "_" + s_dim + "_B" + str(args.B)
#             + "_" + ghash
#         )
#         if args.comment:
#             log_dir += "_" + args.comment
#     else:
#         log_dir = os.path.join(
#             args.runs_dir,
#             args.resume_name
#         )