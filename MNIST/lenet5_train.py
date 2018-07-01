#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:03:51 2018

@author: zhong
"""

import os
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta
import models
import tools
from tensorflow.examples.tutorials.mnist import input_data

num_classes = 10                                                     #### notice
batch_size = 100
max_step = 50000
learning_rate = 1e-4

model_dir = './_lenet5_model/'
logs_train_dir = './logs/_lenet5_train/'
logs_val_dir = './logs/_lenet5_val/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)


def train_running():
    
    with tf.Graph().as_default():
    
        with tf.name_scope('input'):
        
            mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

        x = tf.placeholder(tf.float32, shape=[None, 784])
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        y_ = tf.placeholder(tf.float32, [None, num_classes])
        keep_prob = tf.placeholder(tf.float32)
    
        model = models.Model(x_reshape, num_classes)
        model.lenet5()
        logits = model.logits
    
        loss = tools.loss(logits, y_)
        regular_loss = tf.add_n(tf.get_collection('loss'))
        loss = loss + 1e-4 * regular_loss
        acc = tools.accuracy(logits, y_)
        train_op = tools.optimize(loss, learning_rate)        
    
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
    
            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
            val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
            
            start_time = time.time()
            print('Training Start...')
            for step in np.arange(max_step):

                tra_images, tra_labels = mnist.train.next_batch(batch_size)
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x: tra_images, y_: tra_labels, keep_prob:0.5})

                if step % 50 == 0:
                    print('Step %d, train loss = %.4f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc))
                    summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels, keep_prob:0.5})
                    train_writer.add_summary(summary_str, step)
#                #
                if step % 200 == 0 or (step + 1) == max_step:
                    val_loss, val_acc = sess.run([loss, acc], 
                                                 feed_dict={x:mnist.validation.images, y_:mnist.validation.labels, keep_prob:1.0})
                    print('**  Step %d, val loss = %.4f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))
                    summary_str = sess.run(summary_op, 
                                           feed_dict={x:mnist.validation.images, y_:mnist.validation.labels, keep_prob:1.0})
                    val_writer.add_summary(summary_str, step)
                    #
                if step % 2000 == 0 or (step + 1) == max_step:
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step+1)
                    
            end_time = time.time()
            time_dif = end_time - start_time
            print('Training end...')
            print('Time usage: ' + str(timedelta(seconds=int(round(time_dif)))))
            
            print('Testing...')
            test_acc = sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}) 
            print('Test accuarcy: %.2f%%' % test_acc)


if __name__ == '__main__':
    train_running()