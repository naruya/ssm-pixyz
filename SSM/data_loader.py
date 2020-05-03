import math
import numpy as np
import torch
import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()


class PushDataLoader:
    def __init__(self, args, split, shuffle):
        SEED = args.seed
        np.random.seed(SEED)

        self.T = args.T
        self.B = args.B
        self.ds, self.info = tfds.load(
            name="bair_robot_pushing_small",
            data_dir=args.data_dir,
            split=split,
            in_memory=False,
            with_info=True,
        )
        self.N = self.info.splits[split].num_examples
        self.L = math.ceil(self.N / self.B)
        if shuffle:
            self.ds = self.ds.shuffle(1024, SEED).batch(self.B)  # TODO: buffersize, interleave()
        else:
            self.ds = self.ds.batch(self.B)  # TODO: buffersize, interleave()
        self.ds = self.ds.map(
            self.func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE).repeat(args.epochs)
        self.ds = tfds.as_numpy(self.ds)
        self.itr = 0

    def func(self, data):
        x = (
            tf.dtypes.cast(
                tf.transpose(data["image_aux1"], [0, 1, 4, 2, 3]), tf.float32
            )
            / 255.0
        )
        a = data["action"]
        _s = np.random.randint(30 - (self.T+1) + 1)
        x_0 = x[:, _s]  # 1 frame for preddicting s_0
        x = x[:, _s+1:_s+1+self.T]
        a = a[:, _s+1:_s+1+self.T]
        return {"x_0": x_0, "x": x, "a": a}

    def __iter__(self):
        return self

    def __next__(self):
        self.itr += 1
        batch = next(self.ds)
        # TODO: use state or not
        x_0, x, a = batch["x_0"], batch["x"], batch["a"]
        return torch.from_numpy(x_0), torch.from_numpy(x), torch.from_numpy(a), self.itr

    def __len__(self):
        return self.L