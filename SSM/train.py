from config import get_args
from tqdm import tqdm
from model import *
from data_loader import PushDataLoader
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torch

PLOT_SCALAR_INTERVAL = 169
TRAIN_INTERVAL = 1352  # 43264 / 32
TEST_INTERVAL = 8  # 256 / 32


def data_loop(epoch, loader, model, T, device, writer=None, train=True):
    for batch in tqdm(loader):
        x, a, itr = batch
        _B = x.size(0)
        x = x.to(device).transpose(0, 1)  # T,B,3,28,28
        a = a.to(device).transpose(0, 1)  # T,B,1
        feed_dict = {"x0": x[0].clone(), "x": x, "a": a}

        loss, omake_dict = model.forward_(feed_dict, epoch, train)

        omake_dict = flat_dict(omake_dict)

        try:
            summ
        except NameError:
            keys = omake_dict.keys()
            summ = dict(zip(keys, [0.] * len(keys)))
        summ = update_summ(summ, omake_dict, _B)

        if train and itr % TRAIN_INTERVAL == 0:
            break
        if not train and itr % TEST_INTERVAL == 0:
            break

    if writer:
        video = model.sample_x(feed_dict)
        summ = write_summ(summ, video, wtiter, loader.N)

if __name__ == "__main__":
    args = get_args()
    device = args.device_ids[0]

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    print(args.log_dir)

    if args.comment == "debug":
        writer = None
    else:
        writer = SummaryWriter(log_dir=args.log_dir)

    if args.model == "SSM":
        model = SSM(args, device)
    elif args.model == "SimpleSSM":
        args.s_dim = args.s_dim[0]
        model = SimpleSSM(args, device)
    else:
        raise NotImplementedError

    if args.resume:
        load_model(model, args.resume_name, args.resume_time)

    train_loader = PushDataLoader("train", args)
    test_loader = PushDataLoader("test", args)

    resume_epoch = 1 if not args.resume else args.resume_epoch

    for epoch in range(resume_epoch, args.epochs + 1):
        print(epoch)
        data_loop(epoch, train_loader, model, args.T, device, writer, train=True)
        data_loop(epoch, test_loader, model, args.T, device, writer, train=False)
        save_model(model, args.log_dir.split("/")[-1])