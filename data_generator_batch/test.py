#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:30:03 2018

@author: zhong
"""

import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""

# params
batch_size = 128
img_h = 60
img_w = 80
dropout_rate = 0.5
num_classes = 6
display_step = 20  # How often we want to write the tf.summary data to disk

# Path
test_file = './test.txt'
model_path = "./models/"
    
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    test_data = ImageDataGenerator(test_file,
                                   mode='inference',
                                   batch_size=batch_size,
                                   num_classes=num_classes,
                                   shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(test_data.data.output_types,
                                       test_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
test_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, [])

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
test_batches_per_epoch = int(np.floor(test_data.data_size/batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained model
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')


    # Validate the model on the entire validation set
    print("Start testing")
    sess.run(test_init_op)
    test_acc = 0.
    test_count = 0
    for _ in range(test_batches_per_epoch):

        img_batch, label_batch = sess.run(next_batch)
        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                            y: label_batch,
                                            keep_prob: 1.})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("Testing Accuracy = {:.2f}%".format(test_acc*100.0))
