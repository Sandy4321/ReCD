from __future__ import absolute_import

import tensorflow as tf

def batch_norm(inputT, is_training, name='BN'):
    is_training = tf.cast(is_training, 'bool')
    return tf.cond(is_training,
        lambda: tf.layers.batch_normalization(inputT, training=True, name=name, reuse=tf.AUTO_REUSE),
        lambda: tf.layers.batch_normalization(inputT, training=False, name=name, reuse=tf.AUTO_REUSE))

def conv2d_bn_relu(inputT, depth, is_training=True,
                   init_kernel=None, kernel_size=(7,7),
                   strides=(1,1), padding='same', relu=True, name=None):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(inputT, depth, kernel_size, strides=strides, padding=padding,
                                                        kernel_initializer=init_kernel)
        x = batch_norm(x, is_training)
        if relu:
            x = tf.nn.relu(x)
        output_shape = x.get_shape().as_list()
        return x, output_shape


class MaxPoolingWithArgmax2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if tf.keras.backend.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(tf.keras.backend.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = tf.cast(argmax, tf.float32)
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        """
        Seen on https://github.com/tensorflow/tensorflow/issues/2169
        Replace with unpool op when/if issue merged
        Add theano backend
        """
        updates, mask = inputs[0], inputs[1]
        with tf.variable_scope(self.name):
            mask = tf.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask, dtype='int32')
            batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
            values = tf.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]

def unpool_with_argmax(updates, mask, k_size=[1, 2, 2, 1], output_shape=None, name=''):
    '''
    Unpooling function based on the implementation by Panaetius at https://github.com/tensorflow/tensorflow/issues/2169

    INPUTS:
    - updates(Tensor): a 4D tensor of shape [batch_size, height, width, num_channels] that represents the input block to be upsampled
    - mask(Tensor): a 4D tensor that represents the argmax values/pooling indices of the previously max-pooled layer
    - k_size(list): a list of values representing the dimensions of the unpooling filter.
    - output_shape(list): a list of values to indicate what the final output shape should be after unpooling
    - name(str): the string name to name your scope

    OUTPUTS:
    - ret(Tensor): the returned 4D tensor that has the shape of output_shape.

    '''
    with tf.variable_scope(name):
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret
