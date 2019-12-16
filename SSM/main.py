# export CUDA_VISIBLE_DEVICES=0

# from trains import Task
# task = Task.init(project_name="kondo_ssm", task_name="ssm_push")

from config import get_args
import time
from tqdm import tqdm
import numpy as np
import torch
from model import SSM
from data_loader import PushDataLoader
from torch.utils.tensorboard import SummaryWriter

args = get_args()
device = "cuda"

model = SSM(args, device)

train_loader = PushDataLoader("train", args)
test_loader = PushDataLoader("test", args)

writer = SummaryWriter()

PLOT_SCALAR_INTERVAL = 13
PLOT_VIDEO_INTERVAL = 169  # 43264 / 32 / 8
TRAIN_INTERVAL = 1352  # 43264 / 32
TEST_INTERVAL = 8  # 256 / 32


def data_loop(epoch, loader, model, T, device, writer, train=False, plot=True):
    mean_loss = 0
    time.sleep(0.5)

    for batch in tqdm(loader):
        x, a, itr = batch
        _B = x.size(0)
        x = x.to(device).transpose(0, 1)  # 30,32,3,28,28
        a = a.to(device).transpose(0, 1)  # 30,32,1

        feed_dict = {"x0": x[0], "x": x, "a": a}  # TODO: .clone()要る?
        if train:
            loss = model.train(feed_dict).item() * _B
        else:
            loss = model.test(feed_dict).item() * _B
        mean_loss += loss

        if train and itr % PLOT_SCALAR_INTERVAL == 0:
            writer.add_scalar("loss/itr_train", loss, itr)
        if train and itr % PLOT_VIDEO_INTERVAL == 0 and plot:
            video = model.sample_video_from_latent_s(batch)
            writer.add_video("video/train", video, itr)
        if train and itr % TRAIN_INTERVAL == 0:
            break
        if not train and itr % TEST_INTERVAL == 0:
            break

    mean_loss /= loader.N
    if train:
        writer.add_scalar("loss/train", mean_loss, epoch)
    else:
        writer.add_scalar("loss/test", mean_loss, epoch)

    if not train and plot:
        video = model.sample_video_from_latent_s(batch)
        writer.add_video("video/test", video, itr)

    # file = "epoch{:03d}-iter{:06d}.pt".format(epoch, itr)
    # TODO save, load


for epoch in range(1, args.epochs + 1):
    print(epoch)
    data_loop(epoch, train_loader, model, args.T, device, writer, train=True, plot=True)
    data_loop(epoch, test_loader, model, args.T, device, writer, train=False, plot=True)