from __future__ import absolute_import

import numpy as np
from scipy.misc import imread
import tensorflow as tf

def median_frequency_balancing(image_files, num_classes=11):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c
    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.
    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    - num_classes(int): the number of classes of pixels in all images
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.
    '''
    class_pixels_dict = {}
    total_class_pixels_dict = {}
    frequency_dict = {}

    for i in xrange(num_classes):
        class_pixels_dict[i] = []
        total_class_pixels_dict[i] = []
        frequency_dict[i] = 0

    for n in xrange(len(image_files)):
        image = imread(image_files[n])
        total_image_pixels = np.prod(image.shape)

        for i in xrange(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)
            class_pixels_dict[i].append(class_frequency)
            if class_frequency != 0.0:
                total_class_pixels_dict[i].append(total_image_pixels)


    for i, class_frequencies in enumerate(class_pixels_dict.values()):
        frequency_dict[i] += sum(class_frequencies)

    for i, total_frequencies in enumerate(total_class_pixels_dict.values()):
        frequency_dict[i] /= sum(total_frequencies)

    median_freq = np.median(frequency_dict.values(), axis=0)

    class_weights = []
    for i, total_frequencies in enumerate(frequency_dict.values()):
        class_weights.append(float(median_freq / total_frequencies))

    return class_weights


def monitor(value, sess, epoch,
            name, logdir, mode='max', metric='f1score'):
    if epoch is 1:
        global saver
        saver = tf.train.Saver()
        global monitor_value
        monitor_value = 0

    if mode not in {'max', 'min'}:
        raise ValueError('mode arg must be max or min.')
    if mode is 'max':
        if value >= monitor_value:
            path = "{}/checkpoints/{}-{}:{}".format(logdir, name, metric, value)
            saver.save(sess, path , global_step=epoch)
            print("Model saved. {}: {} --> {}".format(metric, monitor_value, value))
            monitor_value = value
    if mode is 'min':
        if value <= monitor_value:
            path = "{}/checkpoints/{}-{}:{}".format(logdir, name, metric, value)
            saver.save(sess, path , global_step=epoch)
            print("Model saved. {}: {} --> {}".format(metric, monitor_value, value))
            monitor_value = value
