from config import get_args
from tqdm import tqdm
from model import *
from data_loader import PushDataLoader
from pixyz_utils import save_model
from torch.utils.tensorboard import SummaryWriter
import torch

PLOT_SCALAR_INTERVAL = 169
TRAIN_INTERVAL = 1352  # 43264 / 32
TEST_INTERVAL = 8  # 256 / 32


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
    mean = torch.tensor([-3.5140e-05,  2.8279e-05,  4.9905e-01,  2.4916e-01], device=device)
    std = torch.tensor([0.0404, 0.0404, 1.1170, 0.8273], device=device)

    summ = dict(zip(model.keys, [0.] * len(model.keys)))

    for batch in tqdm(loader):
        x, a, itr = batch
        _B = x.size(0)
        x = x.to(device).transpose(0, 1)  # T,B,3,28,28
        x = x.float() / 255.
        a = a.to(device).transpose(0, 1)  # T,B,1
        a = a.sub_(mean).div_(std)

        feed_dict = {"x0": x[0].clone(), "x": x, "a": a}
        if train:
            loss, omake_dict = model.train_(feed_dict, epoch)
        else:
            loss, omake_dict = model.test_(feed_dict, epoch)
        for k in summ.keys():
            v = omake_dict[k]
            summ[k] += v * _B
            if train and itr % PLOT_SCALAR_INTERVAL == 0:
                print(k, v)
                if writer:
                    writer.add_scalar("itr/train_" + k, v, itr)

        if train and itr % TRAIN_INTERVAL == 0:
            break
        if not train and itr % TEST_INTERVAL == 0:
            break

    print("loss:", summ["loss"] / loader.N)

    if writer:
        for k, v in summ.items():
            v = v / loader.N
            summ[k] = v
            writer.add_scalar("epoch/" + prefix + k, v, epoch)
        video = model.sample_x(feed_dict)
        writer.add_video("epoch/" + prefix + "video", video, epoch)

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
    logzero.loglevel(20)
    logzero.logfile(os.path.join("logzero", args.timestamp + ".txt"), loglevel=20)
    logger.info("ghash: " + args.ghash)
    logger.info("command: " + str(sys.argv))
    logger.info(args)


    if args.comment == "debug":
        writer = None
    else:
        writer = SummaryWriter(log_dir=args.log_dir)

    if args.model == "SSM11":
        model = SSM(args, device, query=["a"], extra=None)
    elif args.model == "SSM12":
        model = SSM(args, device, query=["a"], extra="ensemble")
    elif args.model == "SSM13":
        model = SSM(args, device, query=["s"], extra=None)
    elif args.model == "SSM14":
        model = SSM(args, device, query=["s"], extra="residual")
    elif args.model == "SSM15":
        model = SSM(args, device, query=["a", "s"], extra=None)
    elif args.model == "SSM16":
        model = SSM(args, device)
    elif args.model == "SimpleSSM":
        args.s_dim = args.s_dim[0]
        model = SimpleSSM(args, device)
    else:
        raise NotImplementedError

    train_loader = PushDataLoader("train", args)
    test_loader = PushDataLoader("test", args)

    for epoch in range(1, args.epochs + 1):
        print(epoch)
        _summ = data_loop(epoch, train_loader, model, args.T, device, writer, train=True)
        summ  = data_loop(epoch, test_loader, model, args.T, device, writer, train=False)
        if epoch % 10 == 0:
            save_model(model)
            slack("Epoch: {} {} {}".format(epoch, str(sys.argv), str(summ)))
