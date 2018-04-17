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

N_CLASSES = 6                                                      #### notice
IMG_W = 80  
IMG_H = 60
RATIO = 0.3
BATCH_SIZE = 128
MIN_AFTER_DEQUENE = 512
MAX_STEP = 100000                                                  #### notice
MODEL_Name = 'AlexNet'
LEARNING_RATE = 1e-3

data_dir = './data/KTH_RGB/'
model_dir = './model/KTH_RGB{}_{}_bn/'.format(MAX_STEP, MODEL_Name)             #### notice
logs_train_dir = './logs/KTH_RGB{}_{}_bn_train/'.format(MAX_STEP, MODEL_Name)   #### notice
logs_val_dir = './logs/KTH_RGB{}_{}_bn_val/'.format(MAX_STEP, MODEL_Name)       #### notice
train_tfrecords_file ='train.tfrecords'
val_tfrecords_file ='val.tfrecords'

if not os.path.exists(train_tfrecords_file):
    input_data.generate_tfrecond(data_dir, RATIO)

def train_running():
    
    with tf.name_scope('input'):
        train_batch, train_label_batch = input_data.read_and_decode(train_tfrecords_file, IMG_W, IMG_H, BATCH_SIZE, MIN_AFTER_DEQUENE)
        val_batch, val_label_batch = input_data.read_and_decode(val_tfrecords_file, IMG_W, IMG_H, BATCH_SIZE, MIN_AFTER_DEQUENE)
    
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    logits = models.AlexNet(x, N_CLASSES)                               #### notice
    loss = tools.loss(logits, y_)  
    acc = tools.accuracy(logits, y_)
    train_op = tools.optimize(loss, LEARNING_RATE)
             
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],feed_dict={x:tra_images, y_:tra_labels})
                
                if step % 50 == 0:
                    print('Step %d, train loss = %.4f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc))
                    summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_:tra_labels})
                    train_writer.add_summary(summary_str, step)
    #                
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc], feed_dict={x:val_images, y_:val_labels})
                    
                    print('**  Step %d, val loss = %.4f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))
                    summary_str = sess.run(summary_op, feed_dict={x:val_images, y_:val_labels})
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
