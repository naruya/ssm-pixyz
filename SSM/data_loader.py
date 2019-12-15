import math
import numpy as np
import torch
import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()


class PushDataLoader:
    def __init__(self, split, args):
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
        self.ds = self.ds.shuffle(256).batch(self.B)
        self.ds = self.ds.map(
            lambda x: {
                "video": tf.dtypes.cast(
                    tf.transpose(x["image_aux1"], [0, 1, 4, 2, 3]), tf.float32
                )[:, : self.T]
                / 255.0,
                "action": x["action"][:, : self.T],
            },
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE).repeat(args.epochs)
        self.ds = tfds.as_numpy(self.ds)
        self.itr = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.itr += 1
        batch = next(self.ds)
        # TODO: use state or not
        video, action = batch["video"], batch["action"]
        return torch.from_numpy(video), torch.from_numpy(action), self.itr

    def __len__(self):
        return self.L