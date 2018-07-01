#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:43:13 2018

@author: zhong
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.layers.python.layers.regularizers import l1_regularizer
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer


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
#        labels = tf.cast(labels, tf.int64)
#        correct = tf.nn.in_top_k(logits, tf.argmax(labels, 1), 1)
#        correct = tf.cast(correct, tf.float32)
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
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


def plot_feature_map(feature_map):
    
    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flat):
        feature_map = np.array(feature_map)
        feature_map = np.squeeze(feature_map)
        img = feature_map[0,:,:,i]
#        ax.imshow(img, interpolation='nearest', cmap='seismic')
        ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def print_all_variables(train_only=True):
    if train_only:
        t_vars = tf.trainable_variables()
        print("Trainable variables:------------------------")
    else:
        t_vars = tf.global_variables()
        print("Global variables:------------------------")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))   


def plot_confusion_matrix(label_true, label_pred, num_classes):

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))