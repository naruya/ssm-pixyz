import os
import random
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader


# faster than `vread` in `skvideo.io`
def vread(path, T=20):
    cap = cv2.VideoCapture(path)
    gif = [cap.read()[1][...,::-1] for i in range(T)]
    gif = np.array(gif)
    cap.release()
    return gif


class TowelDataset(Dataset):
    def __init__(self, data_dir, mode, T):
        self.actions = np.load(os.path.join(
            data_dir, 'towel_pick_30k', 'npy', mode + '.npy'))
        print("mean:", self.actions.mean(axis=(0,1)))
        print("std:", self.actions.std(axis=(0,1)))
        self.image_paths = sorted(glob.glob(os.path.join(
            data_dir, 'towel_pick_30k', 'gif', mode, '*')))
        self.T = T

        print(len(self.image_paths), self.actions.shape)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        a = self.actions[idx]
        x = vread(self.image_paths[idx])
        x = np.transpose(x, [0,3,1,2])

        _s = np.random.randint(20 - (self.T+1) + 1)
        x_0 = x[_s]
        x = x[_s+1:_s+1+self.T]
        a = a[_s+1:_s+1+self.T]
        return x_0, x, a


class TowelDataLoader(DataLoader):
    def __init__(self, mode, args):
        SEED = args.seed
        np.random.seed(SEED)

        dataset = TowelDataset(data_dir=args.data_dir, mode=mode, T=args.T)
        super(TowelDataLoader, self).__init__(dataset,
                                              batch_size=args.B,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=4)