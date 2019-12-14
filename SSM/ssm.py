# from trains import Task
# task = Task.init(project_name="kondo_ssm", task_name="ssm_push")

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

train_loader = PushDataLoader("~/tensorflow_datasets/",
                              "train", B, args.epochs)
test_loader = PushDataLoader("~/tensorflow_datasets/",
                             "test", B, args.epochs)

writer = SummaryWriter()

for epoch in range(1, args.epochs + 1):
    train_loss = data_loop(epoch, train_loader, model, device, writer,
                           train=True, plot=True)
    test_loss = data_loop(epoch, test_loader, model, device, writer,
                          train=False, plot=True)