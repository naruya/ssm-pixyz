from trains import Task

task = Task.init(project_name="kondo_ssm", task_name="ssm_push")

from config import get_args
import torch
from model import SSM
from data_loader import PushDataLoader
from utils import data_loop, plot_video
from torch.utils.tensorboard import SummaryWriter

args = get_args()
h_dim, s_dim, a_dim = args.h_dim, args.s_dim, args.a_dim
B, T = args.B, args.T
device = "cuda"

model = SSM(h_dim, s_dim, a_dim, T, device)

train_loader = PushDataLoader("~/tensorflow_datasets/", "train", B, 2)
test_loader = PushDataLoader("~/tensorflow_datasets/", "test", B, 2)

writer = SummaryWriter()

path = "logs/figure/epoch{:04d}-{}-{}.png".format(0, 0.0, 0.0)
plot_video(
    model.sample_video_from_latent_s(train_loader), writer, 0, path=path, show=False
)

for epoch in range(1, args.epochs + 1):
    train_loss = data_loop(epoch, train_loader, model, device, train=True)
    test_loss = data_loop(epoch, test_loader, model, device, train=False)
    path = "logs/figure/epoch{:04d}-{}-{}.png".format(epoch, train_loss, test_loss)
    plot_video(
        model.sample_video_from_latent_s(train_loader),
        writer,
        epoch,
        path=path,
        show=False,
    )