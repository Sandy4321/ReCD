from __future__ import absolute_import

import tensorflow as tf

from layers import conv2d_bn_relu, unpool_with_argmax

class CDNet(object):
    def __init__(self, num_class=12, is_training=True, init_kernel=None):
        self._num_class = num_class
        self._is_training = is_training
        self._init_kernel = init_kernel

    def __encoder(self, x, reuse):
        with tf.variable_scope('encoder', reuse=reuse):
            with tf.variable_scope('block0'):
                x, self.__input_shape_pool0 = conv2d_bn_relu(x, 64, self._is_training, self._init_kernel, name='conv2d_bn_relu')
                pool0, self.__pool0_indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME')
            with tf.variable_scope('block1'):
                x, self.__input_shape_pool1 = conv2d_bn_relu(pool0, 64, self._is_training, self._init_kernel, name='conv2d_bn_relu')
                pool1, self.__pool1_indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME')
            with tf.variable_scope('block2'):
                x, self.__input_shape_pool2 = conv2d_bn_relu(pool1, 64, self._is_training, self._init_kernel, name='conv2d_bn_relu')
                pool2, self.__pool2_indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME')
            with tf.variable_scope('block3'):
                x, self.__input_shape_pool3 = conv2d_bn_relu(pool2, 64, self._is_training, self._init_kernel, name='conv2d_bn_relu')
                pool3, self.__pool3_indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME')
                return pool3

    def __decoder(self, x, reuse):
        with tf.variable_scope('decoder', reuse=reuse):
            with tf.variable_scope('block3'):
                unpool3 = unpool_with_argmax(x, self.__pool3_indices,
                                        output_shape=self.__input_shape_pool3,
                                        name='unpool_3')
                x, _ = conv2d_bn_relu(unpool3, 64, self._is_training, self._init_kernel, name='conv2d_bn_relu')
            with tf.variable_scope('block2'):
                unpool2 = unpool_with_argmax(x, self.__pool2_indices,
                                        output_shape=self.__input_shape_pool2,
                                        name='unpool_2')
                x, _ = conv2d_bn_relu(unpool2, 64, self._is_training, self._init_kernel, name='conv2d_bn_relu')
            with tf.variable_scope('block1'):
                unpool1 = unpool_with_argmax(x, self.__pool1_indices,
                                        output_shape=self.__input_shape_pool1,
                                        name='unpool_1')
                x, _ = conv2d_bn_relu(unpool1, 64, self._is_training, self._init_kernel, name='conv2d_bn_relu')
            with tf.variable_scope('block0'):
                unpool0 = unpool_with_argmax(x, self.__pool0_indices,
                                        output_shape=self.__input_shape_pool0,
                                        name='unpool_0')
                x, _ = conv2d_bn_relu(unpool0, 64, self._is_training, self._init_kernel, name='conv2d_bn_relu')


            x, _ = conv2d_bn_relu(x, self._num_class, self._is_training, self._init_kernel,
                                    kernel_size=(1,1), relu=False, name='conv2d_bn_relu_1x1')
            return x

    def __classifier(self, x, reuse):
        with tf.variable_scope('classifier', reuse=reuse):
            logits = tf.nn.softmax(x)

        return logits

    def forward(self, x, reuse=None):
        enc_output = self.__encoder(x, reuse=reuse)
        dec_output = self.__decoder(enc_output, reuse=reuse)
        return self.__classifier(dec_output, reuse=reuse), dec_output

    def backward(self, loss, lr=0.001):
        with tf.name_scope('BackProp'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                train_step = optimizer.minimize(loss)

        return train_step
