import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch


SUMMARY_INTERVAL = 13 # 169 # 43264 / 32 / 8
TRAIN_INTERVAL = 1352 # 43264 / 32
TEST_INTERVAL = 8 # 256 / 32


def data_loop(epoch, loader, model, device, writer, train=False, plot=True):
    mean_loss = 0
    time.sleep(0.5)
    flag_end_epoch = False  # for tqdm normal termination

    for batch in tqdm(loader):
        if flag_end_epoch:
            break
        x, a, itr = batch
        _B = x.size()[0]  # TODO size(0)
        x = x.to(device)  # 32,30,3,28,28
        a = a.to(device)  # 32,30,1
        x = x.transpose(0, 1)  # 30,32,3,28,28
        a = a.transpose(0, 1)  # 30,32,1

        # s_prev = torch.zeros(_B, model.s_dim).to(device)
        # feed_dict = {"x": x, "s_prev": s_prev, "a": a}
        feed_dict = {"x0": x[0], "x": x, "a": a}
        if train:
            loss = model.train(feed_dict).item() * _B
        else:
            loss = model.test(feed_dict).item() * _B
        mean_loss += loss

        if train and itr % SUMMARY_INTERVAL == 0 and plot:
            writer.add_scalar('loss/itr_train', loss, itr)
            path = "logs/figure/train_epoch{:04d}-itr{:04d}.png".format(epoch, itr)
            plot_video(
                model.sample_video_from_latent_s(batch), writer, itr, path, train=True,
            )
        if train and itr % TRAIN_INTERVAL == 0:
            flag_end_epoch = True
        if not train and itr % TEST_INTERVAL == 0:
            flag_end_epoch = True

    if not train and plot:
        path = "logs/figure/test_epoch{:04d}-itr{:04d}.png".format(epoch, itr)
        plot_video(
            model.sample_video_from_latent_s(batch), writer, itr, path, train=False
        )

    mean_loss /= loader.num_examples
    if train:
        writer.add_scalar('loss/train', mean_loss, epoch)
        print("Epoch: {} Train loss: {:.4f}".format(epoch, mean_loss))
    else:
        writer.add_scalar('loss/test', mean_loss, epoch)
        print("Epoch: {} Test loss: {:.4f}".format(epoch, mean_loss))

    return mean_loss


def plot_video(video, writer, itr, path, show=False, train=True):
    # 32, 30, 3, 64, 64
    if isinstance(video, np.ndarray):
        assert False
    elif isinstance(video, torch.Tensor):
        if writer:
            if train:
                writer.add_video("video/train", video, itr)  # NTCHW
            else:
                writer.add_video("video/test", video, itr)  # NTCHW
        video = (
            video.cpu().detach().numpy().astype(np.float).transpose(0, 1, 3, 4, 2)[0]
        )

    plt.figure(figsize=(10, 3))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(video[i])
    if path:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()