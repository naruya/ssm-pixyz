import os
import gc
import psutil
from PIL import Image

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3
BATCH_SIZE = 25
IMG_HEIGHT = 64
IMG_WIDTH = 64
SEQ_LEN = 25


def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0]/2.**30
    print('memory use:', memory_use)


def read_and_decode(filename_queue, reader):
    """
    :param filename_queue:
    :return: list of [numpy arr (25, 64, 64, 3)]
    """
    _, serialized_example = reader.read(filename_queue)  # 次のレコードの key と value が返ってきます
    image_seq = []

    for i in range(SEQ_LEN):
        image_name = 'move/' + str(i) + '/image/encoded'
        features = {image_name: tf.FixedLenFeature([1], tf.string)}
        features = tf.parse_single_example(serialized_example, features=features)

        image_buffer = tf.reshape(features[image_name], shape=[])
        image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
        image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

        image = tf.reshape(image, [1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
        image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
        image_seq.append(image)
    image_seq = tf.concat(image_seq, 0)

    robot_name_key = 'robot/name'
    features = {robot_name_key: tf.FixedLenFeature([], tf.string)}
    robot_name = tf.parse_single_example(serialized_example, features=features)[robot_name_key]
    return image_seq, robot_name


def build_image_input(data_dir, save_dir, pic_dir):
    """Create input tfrecord tensors.
    Args:
      data_dir: 'push/push_train', 'push/push_testnovel', 'push/push_testseen'
      save_dir:
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pic_dir, exist_ok=True)
    filenames = gfile.Glob(os.path.join(data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=1)
    reader = tf.TFRecordReader()
    video_, robot_ = read_and_decode(filename_queue, reader)  # キューからデコードされた画像データとラベルを取得する処理を定義

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初期化
        sess.run(tf.local_variables_initializer())

        try:
            coord = tf.train.Coordinator()  # スレッドのコーディネーター
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # グラフで収集されたすべてのキューランナーを開始

            i = 0
            while not coord.should_stop():
                image_seq, robot_name = sess.run([video_, robot_])  # 次のキューから画像データを取得
                image_seq = image_seq.astype(np.uint8)
                save_filename = str(int(robot_name)) + '_' + str(i)
                np.save(os.path.join(save_dir, save_filename + '.npy'), image_seq)
                Image.fromarray(image_seq[0]).save(os.path.join(pic_dir, save_filename + '.png'))
                i += 1

        except tf.errors.OutOfRangeError:
            pass

        except:
            pass

        finally:
            coord.request_stop()  # すべてのスレッドが停止するように要求
            coord.join(threads)  # スレッドが終了するのを待つ
    gc.collect()
    memory()

    return


if __name__ == '__main__':
    data_dirs = ['push/push_train', 'push/push_testnovel', 'push/push_testseen']
    save_dirs = ['npy/push_train', 'npy/push_testnovel', 'npy/push_testseen']
    pic_dirs = ['pic/push_train', 'pic/push_testnovel', 'pic/push_testseen']
    for data_dir, save_dir, pic_dir in zip(data_dirs, save_dirs, pic_dirs):
        build_image_input(data_dir, save_dir, pic_dir)