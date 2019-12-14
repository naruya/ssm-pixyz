import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch


def data_loop(epoch, loader, model, device, train=False):
    mean_loss = 0
    time.sleep(0.5)
    flag_end_epoch = False  # for tqdm normal termination
    for x, a, end_epoch in tqdm(loader):
        x = x.to(device)  # 128,30,3,28,28
        a = a.to(device)  # 128,30,1
        _B = x.size()[0]
        x = x.transpose(0, 1)          # 30,128,3,28,28
        a = a.transpose(0, 1)          # 30,128,1

        s_prev = torch.zeros(_B, model.s_dim).to(device)
        feed_dict = {'x': x, 's_prev': s_prev, 'a': a}
        if train:
            mean_loss += model.train(feed_dict).item() * _B
        else:
            mean_loss += model.test(feed_dict).item() * _B
        if end_epoch:
            flag_end_epoch = True
    mean_loss /= loader.num_examples
    if train:
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, mean_loss))
    else:
        print('Epoch: {} Test loss: {:.4f}'.format(epoch, mean_loss))
    return mean_loss


def plot_video(video, writer, epoch, path=None, show=False):
    # 32, 30, 3, 64, 64
    if isinstance(video, np.ndarray):
        pass
    elif isinstance(video, torch.Tensor):
        if writer:
            writer.add_video('video', video, epoch)  # NTCHW
        video = video.cpu().detach().numpy().astype(
            np.float).transpose(0, 1, 3, 4, 2)[0][-10:]

    plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(video[i])
    if path:
        plt.savefig(path)
    if show:
        plt.show()