import numpy as np
import torch
import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()
import math


class PushDataLoader:
    def __init__(self, path, split, B, epochs):
        self.ds, self.info = tfds.load(
            name="bair_robot_pushing_small",
            data_dir=path,
            split=split,
            in_memory=False,
            with_info=True,
        )
        self.N = self.info.splits[split].num_examples
        self.L = math.ceil(self.N / B)
        self.ds = self.ds.shuffle(1024).batch(B)
        # B,T,W,H,C => B,T,C,W,H
        self.ds = self.ds.map(
            self.func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE).repeat(epochs)
        self.ds = tfds.as_numpy(self.ds)

    def func(self, data):
        _s = np.random.randint(30)
        x = tf.dtypes.cast(
            tf.transpose(data["image_aux1"][:, _s], [0, 3, 1, 2]), tf.float32
        )
        return {"image_aux1": x / 255.0}

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.ds)
        # TODO: use state or not
        video = batch["image_aux1"]
        return torch.from_numpy(video)

    def __len__(self):
        return self.L