from __future__ import absolute_import

import tensorflow as tf

from layers import MaxPoolingWithArgmax2D

class eLSTM(object):
    def __init__(self, num_class=12, is_training=True, init_kernel=None):
        self._num_class = num_class
        self._is_training = is_training
        self._init_kernel = init_kernel

    def initial_block(self, inputT, filters=13, size=(3, 3), strides=(2, 2), reuse=None):
        with tf.variable_scope('initial_block', reuse=reuse):
            conv = tf.keras.layers.Conv2D(filters, size, padding='same', strides=strides)(inputT)
            max_pool, indices = MaxPoolingWithArgmax2D()(inputT)
            merged = tf.keras.layers.concatenate([conv, max_pool], axis=3)
            return merged, indices

    def encoder_bottleneck(self, inputT, filters, asymmetric=0,
                   dilated=0, downsample=False, dropout_rate=0.1, name=None, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            # main branch
            internal = filters // 2

            # 1x1
            input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
            x = tf.keras.layers.Conv2D(internal, (input_stride, input_stride),
                             strides=(input_stride, input_stride), use_bias=False)(inputT)
            # Batch normalization + PReLU
            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
            x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

            # conv
            if not asymmetric and not dilated:
                x = tf.keras.layers.Conv2D(internal, (3, 3), padding='same')(x)
            elif asymmetric:
                x = tf.keras.layers.Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(x)
                x = tf.keras.layers.Conv2D(internal, (asymmetric, 1), padding='same')(x)
            elif dilated:
                x = tf.keras.layers.Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(x)
            else:
                raise(Exception('You shouldn\'t be here'))

            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
            x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

            # 1x1
            x = tf.keras.layers.Conv2D(filters, (1, 1), use_bias=False)(x)

            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
            x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)

            # other branch
            other = inputT
            if downsample:
                other, indices = MaxPoolingWithArgmax2D(padding='valid')(other)

                pad_feature_maps = filters - inputT.get_shape().as_list()[3]
                if pad_feature_maps > 0:
                    other = tf.keras.layers.Permute((1, 3, 2))(other)
                    tb_pad = (0, 0)
                    lr_pad = (0, pad_feature_maps)
                    other = tf.keras.layers.ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
                    other = tf.keras.layers.Permute((1, 3, 2))(other)
                else:
                    other = tf.keras.layers.Conv2D(filters, (1, 1), use_bias=False)(other)

            x = tf.keras.layers.add([x, other])
            x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
            if downsample:
                return x, indices
            else:
                return x

    def decoder_bottleneck(self, inputT, filters, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            internal = filters // 2

            x = tf.keras.layers.Conv2D(internal, (1, 1), use_bias=False)(inputT)
            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
            x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

            x = tf.keras.layers.Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
            x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

            x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
            x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

            return x

    def encoder(self, inputT, dropout_rate=0.01, reuse=None):
        with tf.variable_scope('encoder', reuse=reuse):
            self.pooling_indices = []
            x, indices_single = self.initial_block(inputT)
            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
            x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
            self.pooling_indices.append(indices_single)

            x, indices_single = self.encoder_bottleneck(x, 32, downsample=True, dropout_rate=dropout_rate, name="bottleneck_1.0")  # bottleneck 1.0
            self.pooling_indices.append(indices_single)
            x = self.encoder_bottleneck(x, 32, dilated=2, name="bottleneck_1.2")
            x = self.encoder_bottleneck(x, 32, asymmetric=5, name="bottleneck_1.3")
            x = self.encoder_bottleneck(x, 32, dilated=4, name="bottleneck_1.4")

            x, indices_single = self.encoder_bottleneck(x, 64, downsample=True, dropout_rate=dropout_rate, name="bottleneck_2.0")  # bottleneck 2.0
            x = self.encoder_bottleneck(x, 64, dilated=2, name="bottleneck_2.1")
            x = self.encoder_bottleneck(x, 64, asymmetric=5, name="bottleneck_2.2")
            x = self.encoder_bottleneck(x, 64, dilated=4, name="bottleneck_2.3")

            x, indices_single = self.encoder_bottleneck(x, 32, downsample=True, dropout_rate=dropout_rate, name="bottleneck_3.0")  # bottleneck 3.0
            x = self.encoder_bottleneck(x, 32, dilated=2, name="bottleneck_3.1")
            x = self.encoder_bottleneck(x, 32, asymmetric=5, name="bottleneck_3.2")
            x = self.encoder_bottleneck(x, 32, dilated=4, name="bottleneck_3.3")

            return x


    def LSTM(self, x, reuse=None):
        with tf.variable_scope('LSTM', reuse=reuse):
            output, state = tf.nn.dynamic_rnn(
                                    tf.contrib.rnn.LSTMCell(5280),
                                    x, dtype=tf.float32)
            batch_size = tf.shape(output)[0]
            max_length = tf.shape(output)[1]
            out_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length
            flat = tf.reshape(output, [-1, out_size])
            relevant = tf.gather(flat, index)
            return relevant

    def decoder(self, inputT, reuse=None):
        with tf.variable_scope('decoder', reuse=reuse):
            x = self.decoder_bottleneck(inputT, 32, name='bottleneck_4.0')
            x = self.decoder_bottleneck(x, 32, name='bottleneck_4.1')
            x = self.decoder_bottleneck(x, 16, name='bottleneck_4.2')

            x = tf.keras.layers.Conv2DTranspose(filters=self._num_class, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
            return x

    def classifier(self, x, reuse=None):
        with tf.variable_scope('classifier', reuse=reuse):
            logits = tf.nn.softmax(x)

        return logits

    def forward(self, inputT, reuse=None):
        enet = tf.reshape(inputT, [20, 180, 240, 3])
        enet = self.encoder(enet, reuse=reuse)
        enc_shape = enet.get_shape().as_list()
        enet = tf.reshape(enet, [10, 2, -1])
        enet = self.LSTM(enet, reuse=reuse)
        enet = tf.reshape(enet, [10, 11, 15, 32])
        enet = self.decoder(enet, reuse=reuse)
        enet = tf.image.resize_images(enet, (180, 240))

        return self.classifier(enet, reuse=reuse), enet


    def backward(self, loss, lr=0.001):
        with tf.name_scope('BackProp'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                train_step = optimizer.minimize(loss)

        return train_step
