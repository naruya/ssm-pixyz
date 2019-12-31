from config import get_args
from tqdm import tqdm
from model import *
from data_loader import PushDataLoader
from torch.utils.tensorboard import SummaryWriter
from pixyz_utils import save_model


PLOT_SCALAR_INTERVAL = 169
TRAIN_INTERVAL = 1352  # 43264 / 32
TEST_INTERVAL = 8  # 256 / 32


def data_loop(epoch, loader, model, T, device, writer=None, train=True):
    mean_loss, mean_loss_ce, mean_loss_kl = 0., 0., 0.

    name = model.__class__.__name__
    prefix = "train" if train else "test"

    for batch in tqdm(loader):
        x, a, itr = batch
        _B = x.size(0)
        x = x.to(device).transpose(0, 1)  # T,B,3,28,28
        a = a.to(device).transpose(0, 1)  # T,B,1

        s0 = model.sample_s0(x[0:1].clone())
        feed_dict = {"s_prev": s0, "x": x, "a": a}

        if name == "SSM3":
            if train:
                loss = model.train(feed_dict).item()
            else:
                loss = model.test(feed_dict).item()
            mean_loss += loss * _B
            if train and writer and itr % PLOT_SCALAR_INTERVAL == 0:
                writer.add_scalar("loss/itr_train", loss, itr)

        elif name == "SSM4":
            if train:
                loss, ce, kl = model._train(feed_dict)
            if train:
                loss, ce, kl = model._test(feed_dict)
            loss, ce, kl = loss.item(), ce.item(), kl.item()
            mean_loss += loss * _B
            mean_loss_ce += ce * _B
            mean_loss_kl += kl * _B
            if train and writer and itr % PLOT_SCALAR_INTERVAL == 0:
                writer.add_scalar("loss/itr_train", loss, itr)
                writer.add_scalar("loss/itr_train_ce", ce, itr)
                writer.add_scalar("loss/itr_train_kl", kl, itr)

        if train and itr % TRAIN_INTERVAL == 0:
            break
        if not train and itr % TEST_INTERVAL == 0:
            break
        break

    if name == "SSM3":
        mean_loss /= loader.N
        video = model.sample_video_from_latent_s(batch)
        if writer:
            writer.add_scalar("loss/" + prefix, mean_loss, epoch)
            writer.add_video("video/" + prefix, video, epoch)
        print(mean_loss)

    elif name == "SSM4":
        mean_loss /= loader.N
        mean_loss_ce /= loader.N
        mean_loss_kl /= loader.N
        video = model.sample_video_from_latent_s(batch)
        if writer:
            writer.add_scalar("loss/" + prefix, mean_loss, epoch)
            writer.add_scalar("loss/" + prefix + "_ce", mean_loss_ce, epoch)
            writer.add_scalar("loss/" + prefix + "_kl", mean_loss_kl, epoch)
            writer.add_video("video/" + prefix, video, epoch)
        print(mean_loss)
        print(mean_loss_ce + mean_loss_kl)


if __name__ == "__main__":
    args = get_args()
    device = args.device_ids[0]
    
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
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
        save_model(model, args.comment)
        break