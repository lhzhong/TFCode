#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:43:13 2018

@author: zhong
"""

import tensorflow as tf
import numpy as np


def losses(logits, labels):
    """Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    """
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)

        # to use this loss fuction, one-hot encoding is needed!
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels,
                                                                name='xentropy_per_example')

        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)

    return loss


def accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        labels = tf.cast(labels, tf.int64)
        correct = tf.nn.in_top_k(logits, tf.argmax(labels, 1), 1)
        correct = tf.cast(correct, tf.float32)
        acc = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', acc)
    return acc


def optimizer(loss, learning_rate):
    """Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    """
    with tf.name_scope('optimizer'):
        optimize = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimize.minimize(loss, global_step=global_step)
    return train_op

