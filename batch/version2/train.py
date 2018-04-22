#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:03:51 2018

@author: zhong
"""

import os
import numpy as np
import tensorflow as tf
import input_data
import models
import tools

N_CLASSES = 51
IMG_W = 120  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 80
BATCH_SIZE = 100
CAPACITY = 2000
MAX_STEP = 100000  # with current parameters, it is suggested to use MAX_STEP>10k
LEARNING_RATE = 1e-3  # with current parameters, it is suggested to use learning rate<0.0001
TRAIN_RATIO = 0.7
data_dir = '/home/zhong/practice/feature_extract/data/KTH_RGB_small/'
model_dir = './model/KTH_RGB6000/'
logs_train_dir = './logs/KTH_RGB6000_train/'
logs_val_dir = './logs/KTH_RGB6000_val/'
train_txt = 'train.txt'
val_txt = 'val.txt'

if not os.path.exists(train_txt):
    input_data.generate_txt(data_dir, TRAIN_RATIO)

if not os.path.exists('./model'):
    os.mkdir('./model')


def train_running():
    with tf.name_scope('input'):

        train_batch, train_label_batch, _ = input_data.get_batch(train_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        val_batch, val_label_batch, _ = input_data.get_batch(val_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    logits = models.VGG16(x, N_CLASSES)
    loss = tools.loss(logits, y_)  
    acc = tools.accuracy(logits, y_)
    train_op = tools.optimize(loss, LEARNING_RATE)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                
                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x: tra_images, y_: tra_labels})
                
                if step % 50 == 0:
                    print('Step %d, train loss = %.4f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc))
                    summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_: tra_labels})
                    train_writer.add_summary(summary_str, step)
    #                
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc], 
                                                 feed_dict={x: val_images, y_: val_labels})
                    
                    print('**  Step %d, val loss = %.4f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))
                    summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels})
                    val_writer.add_summary(summary_str, step)  
    #                                
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)


if __name__ == '__main__':
    train_running()
