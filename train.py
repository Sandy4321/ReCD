from __future__ import absolute_import

import tensorflow as tf
import time
import numpy as np
import shutil
import os, sys
import matplotlib.pyplot as plt

from dataset import Dataset
from models import CDNet, eLSTM
from metrics import cross_entropy, precision, recall, f1score
from utils import median_frequency_balancing, monitor


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model', 'CDNet',
                           """Model to train""")
tf.app.flags.DEFINE_string('train_list_file', 'data/train_split.txt',
                           """File where to find a list of all images for training""")
tf.app.flags.DEFINE_string('val_list_file', 'data/test_split.txt',
                           """File where to find a list of all images for validation""")
tf.app.flags.DEFINE_string('logdir', './logs',
                           """File where to find a list of all images for validation""")
tf.app.flags.DEFINE_integer('num_epochs', 150,
                            """Number of epochs to run training.""")
tf.app.flags.DEFINE_integer('num_class', 11,
                            """Number of classes to perform the training step.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of input image for each training batch.""")
tf.app.flags.DEFINE_integer('val_batch_size', 10,
                            """Number of input image for each validation batch.""")
tf.app.flags.DEFINE_integer('buffer_size', 10000,
                            """Size of buffer for Dataset shuffle method.""")

tf.app.flags.DEFINE_float('lr', 0.1, """Leaning rate parameter.""")
tf.app.flags.DEFINE_integer('img_height', 180, """The height for input image.""")
tf.app.flags.DEFINE_integer('img_width', 240, """The width for input image.""")


tf.logging.set_verbosity(tf.logging.INFO)


def train():
  """Train SEGNET with CAMVID dataset"""
  with tf.Graph().as_default():

    train_dataset = Dataset(FLAGS.train_list_file)
    train_dataset.build(num_class=FLAGS.num_class,
                        height=FLAGS.img_height,
                        width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        num_epochs=FLAGS.num_epochs,
                        shuffle=FLAGS.buffer_size)

    train_x1, train_x2, train_y, train_y_ohe = train_dataset.get_next()
    train_batches_per_epoch = len(train_dataset) / FLAGS.batch_size

    val_dataset = Dataset(FLAGS.val_list_file)
    val_dataset.build(num_class=FLAGS.num_class,
                        height=FLAGS.img_height,
                        width=FLAGS.img_width,
                        batch_size=FLAGS.val_batch_size,
                        num_epochs=FLAGS.num_epochs,
                        shuffle=FLAGS.buffer_size)

    val_x1, val_x2, val_y, val_y_ohe = val_dataset.get_next()
    val_batches_per_epoch = len(val_dataset) / FLAGS.val_batch_size

    tf.logging.info(' Model: {}'
                        .format(FLAGS.model))
    tf.logging.info(' Number of training examples: {}'
                        .format(len(train_dataset)))
    tf.logging.info(' Number of validation examples: {}'
                        .format(len(val_dataset)))
    tf.logging.info(' Number of batches per epoch: {}'
                        .format(train_batches_per_epoch))
    tf.logging.info(' Number of validation batches per epoch: {}'
                        .format(val_batches_per_epoch))

    #Median frequency balancing class_weights
    print("Compute Median Frequency Balancing")
    #class_weights = median_frequency_balancing(train_dataset.y)
    class_weights = [0.027638146768482457, 1.9435447608367058, 1.0, 0.2272722416661377, 15.64842042833608, 0.286085652525553, 1.5602306923731635, 0.8940138739340321, 1.4694906910406547, 6.248954806477947, 0.287822264363265]
    print class_weights

    if FLAGS.model == 'CDNet':
        train_inputs = tf.concat([train_x1, train_x2], axis=-1)
        val_inputs = tf.concat([val_x1, val_x2], axis=-1)

        model = CDNet(num_class=FLAGS.num_class,
                      is_training=True,
                      init_kernel=tf.glorot_normal_initializer())
    elif FLAGS.model == 'eLSTM':
        train_inputs = tf.transpose(tf.stack([train_x1, train_x2]), [1, 0, 2, 3, 4])
        val_inputs = tf.transpose(tf.stack([val_x1, val_x2]), [1, 0, 2, 3, 4])

        model = eLSTM(num_class=FLAGS.num_class,
                      is_training=True,
                      init_kernel=tf.glorot_normal_initializer())
    else:
        raise ValueError('No Model found!')


    train_prob, train_logits = model.forward(train_inputs)
    train_loss = cross_entropy(train_logits, train_y_ohe, class_weights=class_weights)
    train_loss_sum = tf.summary.scalar('loss', train_loss)

    #BackPrpagation
    train_op = model.backward(train_loss)

    train_precision, train_precision_stream = precision(train_prob, train_y)
    train_recall, train_recall_stream = recall(train_prob, train_y)
    train_stream_op = tf.group(train_precision_stream, train_recall_stream)

    train_f1score = f1score(train_precision, train_recall)
    train_f1score_sum = tf.summary.scalar('f1score', train_f1score)

    train_summary_op = tf.summary.merge([train_loss_sum,train_f1score_sum])

    val_prob, val_logits = model.forward(val_inputs, reuse=True)
    val_loss = cross_entropy(val_logits, val_y_ohe)
    val_loss_sum = tf.summary.scalar('loss', val_loss)

    val_precision, val_precision_stream = precision(val_prob, val_y)
    val_recall, val_recall_stream = recall(val_prob, val_y)
    val_stream_op = tf.group(val_precision_stream, val_recall_stream)

    val_f1score = f1score(val_precision, val_recall)
    val_f1score_sum = tf.summary.scalar('f1score', val_f1score)

    x1_sum = tf.summary.image('X1', val_x1, max_outputs=3)
    x2_sum = tf.summary.image('X2', val_x2, max_outputs=3)
    gt = tf.cast(val_y, dtype=tf.float32)
    gt_sum = tf.summary.image('GT', gt, max_outputs=3)
    predictions = tf.argmax(val_prob, -1)
    predictions = tf.cast(predictions, dtype=tf.float32)
    predictions = tf.reshape(predictions, shape=[-1, FLAGS.img_height, FLAGS.img_width, 1])
    pred_sum = tf.summary.image('Prediction', predictions, max_outputs=3)

    val_summary_op = tf.summary.merge([val_loss_sum,val_f1score_sum, x1_sum, x2_sum, gt_sum, pred_sum])

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()


    def train_step(step):
        _,  _, _loss, _summary = sess.run([train_op, train_stream_op, train_loss, train_summary_op], feed_dict={tf.keras.backend.learning_phase(): 1})
        _f1score = sess.run(train_f1score)

        return _loss, _f1score, _summary

    def validation_step(step):
        _, _loss, _summary = sess.run([val_stream_op, val_loss, val_summary_op], feed_dict={tf.keras.backend.learning_phase(): 0})
        _f1score = sess.run(val_f1score)

        return _loss, _f1score, _summary

    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_local)
        sess.run(train_dataset.init())
        sess.run(val_dataset.init())

        if os.path.exists(FLAGS.logdir + '/' + FLAGS.model):
            shutil.rmtree(FLAGS.logdir + '/' + FLAGS.model)

        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + FLAGS.model + '/train', graph=tf.get_default_graph())
        val_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + FLAGS.model + '/val')

        for epoch in xrange(1, FLAGS.num_epochs + 1):
            print("Epoch:({}/{})".format(epoch, FLAGS.num_epochs))
            progbar = tf.keras.utils.Progbar(target=train_batches_per_epoch)
            print("Training")
            for step in xrange(1, train_batches_per_epoch + 1):
                try:
                    _loss, _f1score, _summary = train_step(step)
                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    sys.exit(0)
                progbar.update(step, [('loss', _loss), ('f1score', _f1score)])
                if step == train_batches_per_epoch:
                    print("Validation")
                    progbar = tf.keras.utils.Progbar(target=val_batches_per_epoch)
                    for val_step in xrange(1, val_batches_per_epoch + 1):
                        _val_loss, _val_f1score, _val_summary = validation_step(val_step)
                        progbar.update(val_step, [('val_loss', _val_loss), ('val_f1score', _val_f1score)])

            monitor(value=_val_f1score, sess=sess, epoch=epoch,
                    name=FLAGS.model, logdir=FLAGS.logdir + '/' + FLAGS.model)
            train_writer.add_summary(_summary, epoch)
            train_writer.flush()
            val_writer.add_summary(_val_summary, epoch)
            val_writer.flush()

        train_writer.close()
        val_writer.close()


def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run()
