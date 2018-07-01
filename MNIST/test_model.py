#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:03:51 2018

@author: zhong
"""

import os
import tensorflow as tf
import models
import tools
from tensorflow.examples.tutorials.mnist import input_data

num_classes = 10                                                     #### notice
model_dir = './lenet5_model/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)


def test_running():
    with tf.Graph().as_default():

        mnist = input_data.read_data_sets('../MNIST_data/',one_hot=True)

        x = tf.placeholder(tf.float32, shape=[None, 784])
        x_reshape = tf.reshape(x, [-1, 28,28,1])
        y_ = tf.placeholder(tf.float32,[None, num_classes])
    
        model = models.Model(x_reshape, num_classes)
        model.lenet5()
        logits = model.logits
    
        acc = tools.accuracy(logits, y_)
    
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
                    
            test_acc = sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) 
            print('test accuarcy: %.2f%%' % (test_acc))               

if __name__ == '__main__':
    test_running()