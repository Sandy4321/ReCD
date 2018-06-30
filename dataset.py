from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import glob
import click
from skimage import io

classcolors = {
    0: [0,0,0], #mask-out
    0: [255, 255, 255], #no-change
    1: [136, 0, 21], #barrier(brown/red)
    2: [237, 28, 36], #bin(red)
    3: [255, 127, 39], #construction-maintenance(orange)
    4: [255, 242, 0], #misc(yellow)
    5: [34, 177, 76], #other-objects(dark green)
    6: [0, 162, 232], #person-cycle(light blue)
    7: [63, 72, 204], #rubbish(navy blue)
    8: [163, 73, 164], #sign(purple)
    9: [255, 174, 201], #traffic-cone(pink)
    10: [181, 230, 29], #vehicle(lime)
}

class Dataset(object):
    def __init__(self, file_path, files_dir='./data/raw'):
        indices = [format(idx, '03d') for idx in np.loadtxt(file_path, dtype=np.uint16)]

        self.x1 = []
        for idx in indices:
            for filename in sorted(glob.glob(os.path.join(files_dir, str(idx), 'RGB', "1*.png"))):
                self.x1.append(filename)

        self.x2 = []
        for idx in indices:
            for filename in sorted(glob.glob(os.path.join(files_dir, str(idx), 'RGB', "2*.png"))):
                self.x2.append(filename)

        self.y = []
        for idx in indices:
            for filename in sorted(glob.glob(os.path.join(files_dir, str(idx), 'GTC', "gt*.png"))):
                self.y.append(filename)

        x1 = tf.data.Dataset.from_tensor_slices(self.x1)
        x2 = tf.data.Dataset.from_tensor_slices(self.x2)
        y = tf.data.Dataset.from_tensor_slices(self.y)
        self.dataset = tf.data.Dataset.zip((x1, x2, y))

    def __len__(self):
        return len(self.y)

    def build(self, num_class=11,
              height=180, width=240,
              batch_size=10, num_epochs=100,
              shuffle=10000, num_parallel_calls=2):
        self._num_class = num_class
        self._height = height
        self._width = width
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._shuffle = shuffle
        self.dataset = self.dataset.shuffle(self._shuffle)
        self.dataset = self.dataset.map(self.__input_parser, num_parallel_calls=num_parallel_calls)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._batch_size))
        self.dataset = self.dataset.repeat(self._num_epochs)
        self._iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                         self.dataset.output_shapes)

    def get_next(self):
        return self._iterator.get_next()

    def init(self):
        return self._iterator.make_initializer(self.dataset)

    def __input_parser(self, x1_path, x2_path, y_path):
        x1_file = tf.read_file(x1_path)
        x2_file = tf.read_file(x2_path)
        y_file = tf.read_file(y_path)

        x1_img = tf.image.decode_png(x1_file, channels=3)
        x2_img = tf.image.decode_png(x2_file, channels=3)
        y_img = tf.image.decode_png(y_file, channels=1)

        x1, x2, y = self.__preprocessing(x1_img, x2_img, y_img)
        x1, x2, y = self.__flip_randomly_left_right(x1, x2, y)

        y_one_hot = tf.one_hot(tf.squeeze(y), self._num_class)

        return x1, x2, y, y_one_hot

    def __preprocessing(self, x1, x2, y):
        if x1.dtype != tf.float32:
            x1 = tf.image.convert_image_dtype(x1, dtype=tf.float32)

        if x2.dtype != tf.float32:
            x2 = tf.image.convert_image_dtype(x2, dtype=tf.float32)

        x1 = tf.image.resize_images(x1, [self._height, self._width])
        x1.set_shape(shape=(self._height, self._width, 3))

        x2 = tf.image.resize_images(x2, [self._height, self._width])
        x2.set_shape(shape=(self._height, self._width, 3))

        y = tf.image.resize_images(y, [self._height, self._width])
        y.set_shape(shape=(self._height, self._width, 1))

        if y.dtype != tf.int64:
            y = tf.cast(y, tf.int64)

        return x1, x2, y

    def __flip_randomly_left_right(self, x1, x2, y):
        # Random variable: two possible outcomes (0 or 1)
        # with a 1 in 2 chance
        random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])


        randomly_flipped_x1 = tf.cond(pred=tf.equal(random_var, 0),
                                      true_fn=lambda: tf.image.flip_left_right(x1),
                                      false_fn=lambda: x1)

        randomly_flipped_x2 = tf.cond(pred=tf.equal(random_var, 0),
                                      true_fn=lambda: tf.image.flip_left_right(x2),
                                      false_fn=lambda: x2)

        randomly_flipped_y = tf.cond(pred=tf.equal(random_var, 0),
                                     true_fn=lambda: tf.image.flip_left_right(y),
                                     false_fn=lambda: y)

        return randomly_flipped_x1, randomly_flipped_x2, randomly_flipped_y

def maskout_to_nochange(arr):
    arr[(arr==[0, 0, 0]).all(2)] = [255, 255, 255]
    return arr

def img_array_to_single_val(arr, color_codes):
    result = np.ndarray(shape=arr.shape[:2], dtype=np.uint32)
    result[:,:] = -1
    for idx, rgb in color_codes.items():
        result[(arr==rgb).all(2)] = idx
    return result

@click.command()
@click.option('--directory', default='./data/raw')
def imreadgtpng(directory):
    for seq in sorted(os.listdir(directory)):
        try:
            seq = int(seq)
        except Exception as e:
            continue
        seq = format(seq, '03d')
        if not os.path.isdir(os.path.join(directory, seq, 'GTC')):
            os.makedirs(os.path.join(directory, seq, 'GTC'))

        for file_path in sorted(glob.glob(os.path.join(directory, seq, 'GT', "gt*.png"))):
            img = io.imread(file_path)
            img = maskout_to_nochange(img)
            img = img_array_to_single_val(img, classcolors)
            io.imsave(file_path.replace("GT","GTC"), img)

if __name__ == '__main__':
    imreadgtpng()
