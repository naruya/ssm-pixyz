# https://github.com/SudeepDasari/visual_foresight/blob/master/examples/dataset_reader.py

import numpy as np
import tensorflow as tf
import pickle as pkl
import tensorflow as tf
import os
import glob
from tensorflow.contrib.training import HParams
import cv2

tf.compat.v1.disable_eager_execution()

def mult_elems(tup):
    prod = 1
    for t in tup:
        prod *= t
    return prod


class BaseVideoDataset:
    MODES = ['train', 'test', 'val']

    def __init__(self, directory, batch_size, hparams_dict=dict()):
        if not os.path.exists(directory):
            raise FileNotFoundError('Base directory {} does not exist'.format(directory))

        self._base_dir = directory
        self._batch_size = batch_size

        # read dataset manifest and initialize hparams
        self._hparams = self._get_default_hparams().override_from_dict(hparams_dict)
        self._read_manifest()

        # initialize batches (one tf Dataset per each mode where batches are drawn from)
        self._initialize_batches()

    def _get_default_hparams(self):
        default_dict = {'shuffle': True,
                        'num_epochs': None,
                        'buffer_size': 512,
                        'compressed': True,
                        'sequence_length':None,  # read from manifest if None
                        }
        return HParams(**default_dict)

    def _parse_record(self, serialized_example):
        def get_feature(manifest_entry):
            shape, dtype = manifest_entry
            if dtype == 'Byte':
                return tf.FixedLenFeature([1], tf.string)
            elif dtype == 'Float':
                return tf.FixedLenFeature([mult_elems(shape)], tf.float32)
            elif dtype == 'Int':
                return tf.FixedLenFeature([mult_elems(shape)], tf.int64)
            raise ValueError('Unknown dtype: {}'.format(dtype))

        def decode_feat(feat, manifest_entry, pad_t=False):
            orig_shape, dtype = list(manifest_entry[0]), manifest_entry[1]
            shape = [s for s in orig_shape]
            if pad_t:
                shape = [1] + shape

            if dtype == 'Byte':
                uint_data = tf.decode_raw(feat, tf.uint8)
                img_flat = tf.reshape(uint_data, shape=[1, mult_elems(shape)])
                image = tf.reshape(img_flat, shape=orig_shape)
                image = tf.reshape(image, shape=shape)
                return image
            elif dtype == 'Float' or dtype == 'Int':
                return tf.reshape(feat, shape=shape)
            raise ValueError('Unknown dtype: {}'.format(dtype))

        features_names = {}
        for k in self._metadata_keys:
            features_names[k] = get_feature(self._metadata_keys[k])
        if self._T > 0:
            # print(self._T)
            for k in self._sequence_keys:
                for t in range(self._T):
                    features_names['{}/{}'.format(t, k)] = get_feature(self._sequence_keys[k])

        feature = tf.parse_single_example(serialized_example, features=features_names)

        return_dict = {}
        if self._T > 0:
            for k in self._sequence_keys:
                k_feats = []
                for t in range(self._T):
                    k_feat = decode_feat(feature['{}/{}'.format(t, k)], self._sequence_keys[k], True)
                    k_feats.append(k_feat)
                return_dict[k] = tf.concat(k_feats, 0)
        for k in self._metadata_keys:
            return_dict[k] = decode_feat(feature[k], self._metadata_keys[k])

        return return_dict

    def _initialize_batches(self):
        self._raw_data = {}
        for m in self.MODES:
            fnames = glob.glob('{}/{}/*.tfrecords'.format(self._base_dir, m))
            if len(fnames) == 0:
                print('Warning dataset does not have files for mode: {}'.format(m))
                continue

            if self._hparams.compressed:
                dataset = tf.data.TFRecordDataset(fnames, buffer_size=self._hparams.buffer_size, compression_type='GZIP')
            else:
                dataset = tf.data.TFRecordDataset(fnames, buffer_size=self._hparams.buffer_size)

            dataset = dataset.map(self._parse_record)

            ########
#             dataset = dataset.repeat(self._hparams.num_epochs)            
#             if self._hparams.shuffle:
#                 dataset = dataset.shuffle(buffer_size=self._hparams.buffer_size)

            ########

            dataset = dataset.batch(self._batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            output_element = {}
            for k in list(next_element.keys()):
                output_element[k] = tf.reshape(next_element[k],
                                               [self._batch_size] + next_element[k].get_shape().as_list()[1:])

            self._raw_data[m] = output_element

    def _map_key(self, dataset_batch, key):
        if key == 'state' or key == 'endeffector_pos':
            return dataset_batch['env/state']
        elif key == 'actions':
            return dataset_batch['policy/actions']
        elif key == 'images':
            imgs, i = [], 0
            while True:
                image_name = 'env/image_view{}/encoded'.format(i)
                if image_name not in dataset_batch:
                    break
                imgs.append(tf.expand_dims(dataset_batch[image_name], 2))
                i += 1
            if i == 0:
                raise ValueError("No image tensors")
            elif i == 1:
                return imgs[0]
            return tf.concat(imgs, 2)

        elif key in dataset_batch:
            return dataset_batch[key]

        raise NotImplementedError('Key {} not present in batch which has keys:\n {}'.format(key,
                                                                                            list(dataset_batch.keys())))

    def get(self, key, mode='train'):
        if mode not in self._raw_data:
            valid_modes = list(self._raw_data.keys())
            raise ValueError('Mode {} not valid! Dataset has following modes: {}'.format(mode, valid_modes))
        dataset_batch = self._raw_data[mode]
        return self._map_key(dataset_batch, key)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise KeyError('Index should be in format: [Key, Mode] or [Key] (assumes default train mode)')
            key, mode = item
            return self.get(key, mode)

        return self.get(item)

    
########

#     def get_iterator(self, item, mode):
#         fnames = glob.glob('{}/{}/*.tfrecords'.format(self._base_dir, mode))
#         if self._hparams.compressed:
#             dataset = tf.data.TFRecordDataset(fnames, buffer_size=self._hparams.buffer_size,
#                                               compression_type='GZIP')
#         else:
#             dataset = tf.data.TFRecordDataset(fnames, buffer_size=self._hparams.buffer_size)

#         def parse_record(ex):
#             return self._parse_record(ex)[item]


#         dataset = dataset.map(parse_record)
#         dataset = dataset.repeat(self._hparams.num_epochs)
#         if self._hparams.shuffle:
#             dataset = dataset.shuffle(buffer_size=self._hparams.buffer_size)
#         dataset = dataset.batch(self._batch_size)
#         iterator = dataset.make_one_shot_iterator()
#         return iterator

    def _read_manifest(self):
        pkl_path = '{}/manifest.pkl'.format(self._base_dir)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError('Manifest not found at {}/manifest.pkl'.format(self._base_dir))

        manifest_dict = pkl.load(open(pkl_path, 'rb'))
        self._sequence_keys = manifest_dict['sequence_data']
        self._metadata_keys = manifest_dict['traj_metadata']
        if self._hparams.sequence_length is None:
            self._T = manifest_dict['T']
        else:
            self._T = self._hparams.sequence_length

    @property
    def T(self):
        return self._T


import moviepy.editor as mpy
def npy_to_gif_resize(npy, filename, size):
    gif = list(npy)
    gif = [cv2.resize(frame, size) for frame in gif]
    clip = mpy.ImageSequenceClip(gif, fps=10)
    clip.write_gif(filename)


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('dataset_path', type=str, help="path to dataset")
# args = parser.parse_args()

dataset_path = "./"

dataset = BaseVideoDataset(dataset_path, 256)     # parses tfrecords automatically
N_dict = {'train': 27733, 'test': 1583, 'val': 1557}

os.makedirs('./data/gif/', exist_ok=True)
os.makedirs('./data/npy/', exist_ok=True)

for mode in ['val', 'test', 'train']:
    os.makedirs('./data/gif/{}'.format(mode), exist_ok=True)

    image_set = dataset['images', mode]             # (batch_size, T, 2, height, width, 3) 2 camera!!!
    action_set = dataset['actions', mode]           # (batch_size, T, adim)
    sess = tf.Session()

    N = 0
    all_actions = []

    while True:
        images, actions = sess.run([image_set, action_set])
        all_actions.extend(actions)

        for i, image in enumerate(images):
            filename = './data/gif/{}/{:05}.gif'.format(mode, N+i)
            npy_to_gif_resize(image[:,0], filename, (64, 64))  # use front camera

        N += images.shape[0]
        if N == N_dict[mode]:
            break

    all_actions = np.array(all_actions)
    print(all_actions.shape)
    np.save('./data/npy/{}.npy'.format(mode), all_actions)