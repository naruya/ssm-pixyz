import numpy as np
import torch
import tensorflow as tf
import tensorflow_datasets as tfds
tf.compat.v1.enable_eager_execution()
import math

class PushDataLoader():
    def __init__(self, path, split, batch_size, epochs):
        self.ds, self.info = tfds.load(name="bair_robot_pushing_small", data_dir=path, split=split, in_memory=False, with_info=True)
        self.num_examples = self.info.splits[split].num_examples
        self.length = math.ceil(self.num_examples / batch_size)
        self.ds = self.ds.repeat(epochs).shuffle(256).batch(batch_size)
        # B,T,W,H,C => B*T,C,W,H
        self.ds = self.ds.map(
            lambda x:{'image_aux1': tf.dtypes.cast(tf.transpose(
                x['image_aux1'], [0,1,4,2,3]), tf.float32) / 255.},
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.ds = tfds.as_numpy(self.ds)
    def __iter__(self):
        return self
    def __next__(self):
        batch = next(self.ds)
        # TODO: use state or not
        video = batch['image_aux1']
        return torch.from_numpy(video)
    def __len__(self):
        return self.length