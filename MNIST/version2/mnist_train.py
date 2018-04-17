#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:51:23 2018

@author: zhong
"""

import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import models
import os
from tensorflow.examples.tutorials.mnist import input_data

img_size = 28
num_channels = 1
num_classes = 10
train_batch_size = 64
num_iterations = 10000

def training():
### Load Data

    data = input_data.read_data_sets('MNIST_data/', one_hot=True)
    data.test.cls = np.argmax(data.test.labels, axis=1)
    
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    
    predict = models.LeNet(x_image, num_classes)
    y_pred = tf.nn.softmax(predict)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    model_dir = './model/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("create the directory: %s\n" % model_dir)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
           
        start_time = time.time()
        for step in range(num_iterations):
    
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)
            feed_dict_train = {x: x_batch,y_true: y_true_batch}
            sess.run(optimizer, feed_dict=feed_dict_train)
    
            if step % 100 == 0:
                acc = sess.run(accuracy, feed_dict=feed_dict_train)
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                print(msg.format(step + 1, acc))
                
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
                                                 
     
if __name__ == '__main__':
    training()
    
            
            
            