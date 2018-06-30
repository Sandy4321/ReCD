from __future__ import absolute_import

import tensorflow as tf

def cross_entropy(logits, labels, class_weights=None):
    '''
    ------------------
    Technical Details
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want.
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.
    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------
    '''
    with tf.variable_scope('Metrics/Loss'):
        if class_weights:
            weights = labels * class_weights
            weights = tf.reduce_sum(weights, 3)
            cross_entropy_mean = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=weights)
        else:
            cross_entropy_mean = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    return cross_entropy_mean


def precision(probs, labels):
    with tf.variable_scope('Metrics/precision'):
        labels = tf.squeeze(labels)
        predictions = tf.argmax(probs, -1)
        precision, precision_stream = tf.metrics.precision(labels, predictions)

    return precision, precision_stream

def recall(probs, labels):
    with tf.variable_scope('Metrics/recall'):
        labels = tf.squeeze(labels)
        predictions = tf.argmax(probs, -1)
        recall, recall_stream = tf.metrics.recall(labels, predictions)

    return recall, recall_stream


def f1score(precision, recall):
    return 2 * precision * recall / (precision + recall)
