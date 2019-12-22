from config import get_args
from tqdm import tqdm
from data_loader import PushDataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *


PLOT_SCALAR_INTERVAL = 13
TRAIN_INTERVAL = 1352  # 43264 / 32
TEST_INTERVAL = 8  # 256 / 32


def data_loop(epoch, loader, model, T, device, writer, train=True):
    mean_loss = 0
    mean_loss_ce = 0
    mean_loss_kl = 0

    name = model.__class__.__name__
    prefix = "train" if train else "test"

    for batch in tqdm(loader):
        x, a, itr = batch
        print(x.shape, a.shape)
        _B = x.size(0)
        x = x.to(device).transpose(0, 1)  # 30,32,3,28,28
        a = a.to(device).transpose(0, 1)  # 30,32,1

        if name == "SSM3" or name == "SSM4":
            s0 = model.sample_s0(x[0:1])
            feed_dict = {"s_prev": s0, "x": x, "a": a}  # TODO: .clone()要る?
        else:
            raise NotImplementedError

        if train:
            loss, ce, kl, total_norm = model.train_(feed_dict, return_total_norm=True)
            loss = loss.item()
            ce = ce.item()
            kl = kl.item()
        else:
            loss, ce, kl = model.test_(feed_dict)
            loss = loss.item()
            ce = ce.item()
            kl = kl.item()

        if train and itr % PLOT_SCALAR_INTERVAL == 0:
            writer.add_scalar("loss/itr_train", loss, itr)
            writer.add_scalar("loss/itr_train_ce", ce, itr)
            writer.add_scalar("loss/itr_train_kl", kl, itr)
            writer.add_scalar("grad_norm/itr_train", total_norm, itr)

        mean_loss += loss * _B
        mean_loss_ce += model.loss_ce.eval(feed_dict).item() * _B
        mean_loss_kl += model.loss_kl.eval(feed_dict).item() * _B

        if train and itr % TRAIN_INTERVAL == 0:
            break
        if not train and itr % TEST_INTERVAL == 0:
            break

    mean_loss /= loader.N
    mean_loss_ce /= loader.N
    mean_loss_kl /= loader.N
    video = model.sample_video_from_latent_s(batch)

    writer.add_scalar("loss/" + prefix, mean_loss, epoch)
    writer.add_scalar("loss/" + prefix + "_ce", mean_loss_ce, epoch)
    writer.add_scalar("loss/" + prefix + "_kl", mean_loss_kl, epoch)
    writer.add_scalar("s0/" + prefix + "_norm_mean", s0.norm(dim=1).mean(), epoch)
    writer.add_scalar("s0/" + prefix + "_norm_std", s0.norm(dim=1).std(), epoch)
    writer.add_video("video/" + prefix, video, epoch)
    print(mean_loss)
    print(mean_loss_ce + mean_loss_kl)


if __name__ == "__main__":
    args = get_args()
    device = args.device_ids[0]

    if args.model == "SSM3":
        model = SSM3(args, device)
    elif args.model == "SSM4":
        model = SSM4(args, device)
    else:
        raise NotImplementedError

    writer = SummaryWriter(log_dir=args.log_dir)

    train_loader = PushDataLoader("train", args)
    test_loader = PushDataLoader("test", args)

    for epoch in range(1, args.epochs + 1):
        print(epoch)
        data_loop(epoch, train_loader, model, args.T, device, writer, train=True)
        data_loop(epoch, test_loader, model, args.T, device, writer, train=False)
        model.save(args.comment)