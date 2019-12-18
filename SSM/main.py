# export CUDA_VISIBLE_DEVICES=0
# python main.py --model SSM1

# from trains import Task
# task = Task.init(project_name="kondo_ssm", task_name="ssm_push")

from config import get_args
import time
from tqdm import tqdm
import numpy as np
import torch
from data_loader import PushDataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pixyz_utils import *
from model import *

args = get_args()
device = "cuda"

if args.model == "SSM1":
    model = SSM1(args, device)
elif args.model == "SSM2":
    model = SSM2(args, device)
elif args.model == "SSM3":
    model = SSM3(args, device)

train_loader = PushDataLoader("train", args)
test_loader = PushDataLoader("test", args)

log_dir = (
    "../runs/"
    + datetime.now().strftime("%b%d_%H-%M-%S")
    + "_"
    + args.model
    + "_"
    + args.comment
)
writer = SummaryWriter(log_dir=log_dir)

PLOT_SCALAR_INTERVAL = 13
PLOT_VIDEO_INTERVAL = 1352  # 43264 / 32 / 8
TRAIN_INTERVAL = 1352  # 43264 / 32
TEST_INTERVAL = 8  # 256 / 32


def data_loop(epoch, loader, model, T, device, writer, comment, train=True):
    mean_loss = 0
    mean_loss_ce = 0
    mean_loss_kl = 0
    time.sleep(0.5)

    name = model.__class__.__name__

    # if train:
    #     model.distributions.train()
    # else:
    #     model.distributions.eval()

    for batch in tqdm(loader):
        x, a, itr = batch
        _B = x.size(0)
        x = x.to(device).transpose(0, 1)  # 30,32,3,28,28
        a = a.to(device).transpose(0, 1)  # 30,32,1

        if name == "SSM1":
            feed_dict = {"x0": x[0], "x": x, "a": a}  # TODO: .clone()要る?
        # elif name == "SSM2":
        #     # s_prev = model.sample_s0(x[0], train=True)
        #     s_prev = model.sample_s0(x[0], train=False)
        #     feed_dict = {"s_prev": s_prev, "x": x, "a": a}
        elif name == "SSM3":
            s0 = model.sample_s0(x[0:1])
            feed_dict = {"s_prev": s0, "x": x, "a": a}

        if train:
            loss = model.train(feed_dict).item() * _B
        else:
            loss = model.test(feed_dict).item() * _B
        mean_loss += loss
        mean_loss_ce += model.loss_ce.eval(feed_dict).item() * _B
        mean_loss_kl += model.loss_kl.eval(feed_dict).item() * _B

        if train and itr % PLOT_SCALAR_INTERVAL == 0:
            writer.add_scalar("loss/itr_train", loss, itr)
        if train and itr % PLOT_VIDEO_INTERVAL == 0:
            video = model.sample_video_from_latent_s(batch)
            writer.add_video("video/itr_train", video, itr)
        if train and itr % TRAIN_INTERVAL == 0:
            break
        if not train and itr % TEST_INTERVAL == 0:
            break
        # print(model.encoder_s0.fc1.weight[:10])

    mean_loss /= loader.N
    mean_loss_ce /= loader.N
    mean_loss_kl /= loader.N
    video = model.sample_video_from_latent_s(batch)

    if train:
        writer.add_scalar("loss/train", mean_loss, epoch)
        writer.add_scalar("loss/train_ce", mean_loss_ce, epoch)
        writer.add_scalar("loss/train_kl", mean_loss_kl, epoch)
        writer.add_scalar("s0/train_norm_mean", s0.norm(dim=1).mean())
        writer.add_scalar("s0/train_norm_std", s0.norm(dim=1).std())
        writer.add_video("video/train", video, epoch)
        save_model(model, comment)

    else:
        writer.add_scalar("loss/test", mean_loss, epoch)
        writer.add_scalar("loss/test_ce", mean_loss_ce, epoch)
        writer.add_scalar("loss/test_kl", mean_loss_kl, epoch)
        writer.add_scalar("s0/test_norm_mean", s0.norm(dim=1).mean())
        writer.add_scalar("s0/test_norm_std", s0.norm(dim=1).std())
        writer.add_video("video/test", video, epoch)


for epoch in range(1, args.epochs + 1):
    print(epoch)
    data_loop(
        epoch, train_loader, model, args.T, device, writer, args.comment, train=True
    )
    data_loop(
        epoch, test_loader, model, args.T, device, writer, args.comment, train=False
    )