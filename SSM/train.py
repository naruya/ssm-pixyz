from config import get_args
from tqdm import tqdm
from model import *
from data_loader import PushDataLoader
from pixyz_utils import save_model
from torch.utils.tensorboard import SummaryWriter


PLOT_SCALAR_INTERVAL = 169
TRAIN_INTERVAL = 1352  # 43264 / 32
TEST_INTERVAL = 8  # 256 / 32


def data_loop(epoch, loader, model, T, device, writer=None, train=True):
    name = model.__class__.__name__
    prefix = "train_" if train else "test_"

    if name == "SSM3":
        mean_loss = 0.
    elif name == "SSM4":
        mean_values = {"loss": 0., "ce": 0., "kl": 0.}

    for batch in tqdm(loader):
        x, a, itr = batch
        _B = x.size(0)
        x = x.to(device).transpose(0, 1)  # T,B,3,28,28
        a = a.to(device).transpose(0, 1)  # T,B,1

        if name in ["SSM3",]:  # pixyz.models.Model
            s0 = model.sample_s0(x[0:1].clone())
            feed_dict = {"s_prev": s0, "x": x, "a": a}
            if train:
                loss = model.train(feed_dict)
            else:
                loss = model.test(feed_dict)
            loss = loss.item()
            mean_loss += loss * _B
            if train and writer and itr % PLOT_SCALAR_INTERVAL == 0:
                writer.add_scalar("itr/train_loss", loss, itr)

        elif name in ["SSM4",]:  # nn.Module
            feed_dict = {"x0": x[0:1], "x": x, "a": a}
            if train:
                loss, omake_dict = model.train_(feed_dict)
            else:
                loss, omake_dict = model.test_(feed_dict)
            for k in mean_values.keys():
                v = omake_dict[k]
                mean_values[k] += v * _B
                if train and writer and itr % PLOT_SCALAR_INTERVAL == 0:
                    writer.add_scalar("itr/train_" + k, v, itr)

        if train and itr % TRAIN_INTERVAL == 0:
            break
        if not train and itr % TEST_INTERVAL == 0:
            break

    if name in ["SSM3",]:
        mean_loss /= loader.N
        print(mean_loss)
        video = model.sample_video_from_latent_s(batch)
        if writer:
            writer.add_scalar("epoch/" + prefix + "loss", mean_loss, epoch)
            writer.add_video("epoch/" + prefix + "video", video, epoch)

    elif name in ["SSM4",]:
        print("loss:", mean_values["loss"] / loader.N)
        if writer:
            for k, v in mean_values.items():
                v = v / loader.N
                writer.add_scalar("epoch/" + prefix + k, v, epoch)
            video = model.sample_video_from_latent_s(batch)
            writer.add_video("epoch/" + prefix + "video", video, epoch)


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