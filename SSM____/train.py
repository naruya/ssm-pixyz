from config import get_args
from tqdm import tqdm
from model import *
from data_loader import TowelDataLoader
from pixyz_utils import save_model
from torch.utils.tensorboard import SummaryWriter
import torch

PLOT_SCALAR_INTERVAL = 169
# TRAIN_INTERVAL = 1352  # 43264 / 32
# TEST_INTERVAL = 8  # 256 / 32


import requests
import json
def slack(text):
    # webhook_url: "https://hooks.slack.com/services/foo/bar/foobar"
    with open(".slack.txt") as f:
        webhook_url = f.read()
    requests.post(webhook_url, data = json.dumps({"text": text}))


def data_loop(epoch, loader, model, T, device, writer=None, train=True):
    name = model.__class__.__name__
    prefix = "train_" if train else "test_"
    # push
    # mean = torch.tensor([-3.5140e-05,  2.8279e-05,  4.9905e-01,  2.4916e-01], device=device)
    # std = torch.tensor([0.0404, 0.0404, 1.1170, 0.8273], device=device)
    # towel
    mean = torch.tensor([0.00025036, 0.00028587, 0.03307046, 0.0008472], device=device)
    std = torch.tensor([0.10005278, 0.10012137, 0.27526397, 0.24198195], device=device)

    summ = dict(zip(model.keys, [0.] * len(model.keys)))
    N = 0

    for batch in tqdm(loader):
        x_0, x, a = batch

        x_0 = x_0.to(device).float() / 255.  # B,3,28,28
        _B = x_0.size(0)
        N += _B
        x = x.to(device).float().transpose(0, 1) / 255. # T,B,3,64,64
        a = a.to(device).transpose(0, 1)  # T,B,1
        a = a.sub_(mean).div_(std)

        feed_dict = {"x0": x_0, "x": x, "a": a}
        if train:
            loss, omake_dict = model.train_(feed_dict, epoch)
        else:
            loss, omake_dict = model.test_(feed_dict, epoch)
        for k in summ.keys():
            v = omake_dict[k]
            summ[k] += v * _B
            # if train and itr % PLOT_SCALAR_INTERVAL == 0:
            #     print(k, v)
            #     if writer:
            #         writer.add_scalar("itr/train_" + k, v, itr)

        # if train and itr % TRAIN_INTERVAL == 0:
        #     break
        # if not train and itr % TEST_INTERVAL == 0:
        #     break

    print("loss:", summ["loss"] / N)

    if writer:
        for k, v in summ.items():
            v = v / N
            summ[k] = v
            writer.add_scalar("epoch/" + prefix + k, v, epoch)
        if epoch % 10 == 0:
            video = model.sample_x(feed_dict)
            writer.add_video("epoch/" + prefix + "x", video, epoch)
            # video = model.sample_dx(feed_dict)
            # writer.add_video("epoch/" + prefix + "dx", video, epoch)

    logger.info("({}) Epoch: {} {}".format(prefix, epoch, summ))
    return summ


if __name__ == "__main__":
    args = get_args()
    device = args.device_ids[0]

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    print(args.log_dir)

    assert args.B == 32, "check INERVALS!"
    import os
    import sys
    import logzero
    from logzero import logger
    assert os.path.isfile(".slack.txt"), "error"
    os.makedirs("./logzero", exist_ok=True)
    logzero.loglevel(20)
    logzero.logfile(os.path.join("logzero", args.timestamp + ".txt"), loglevel=20)
    logger.info("ghash: " + args.ghash)
    logger.info("command: " + str(sys.argv))
    logger.info(args)

    if args.comment == "debug":
        writer = None
    else:
        writer = SummaryWriter(log_dir=args.log_dir)

    model = SSM(args, device)

    train_loader = TowelDataLoader("train", args)
    test_loader = TowelDataLoader("test", args)

    for epoch in range(1, args.epochs + 1):
        print(epoch)
        _summ = data_loop(epoch, train_loader, model, args.T, device, writer, train=True)
        summ  = data_loop(epoch, test_loader, model, args.T, device, writer, train=False)
        if epoch % 10 == 0:
            save_model(model)
            slack("{} Epoch: {} {} {}".format(args.ghash, epoch, str(sys.argv), str(summ)))
