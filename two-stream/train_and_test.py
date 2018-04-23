#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:41:02 2018

@author: cpss
"""

import os
import numpy as np
import tensorflow as tf
import input_data
import models
import tools
import math

N_CLASSES = 6
IMG_W = 120  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 120
TRAIN_RATIO = 0.7 # take 20% of dataset as validation data 
BATCH_SIZE = 128
CAPACITY = 2000
MAX_STEP = 30000 
LEARNING_RATE = 1e-3 

data_dir1 = './data/KTH_RGB/'
data_dir2 = './data/KTH_Flow/'
model_dir = './model/KTH_twostream/'
logs_train_dir = './logs/KTH_train/'
logs_val_dir = './logs/KTH_val/'
train_txt = 'train.txt'
val_txt = 'val.txt'

if not os.path.exists(train_txt):
    input_data.generate_txt(data_dir1, data_dir2, TRAIN_RATIO)

if not os.path.exists('./model'):
    os.mkdir('./model')

def train_and_test():
    
    with tf.name_scope('input'):
        train_RGB_batch, train_FLOW_batch, train_label_batch, _ = input_data.get_batch(train_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        val_RGB_batch, val_FLOW_batch, val_label_batch, n_val = input_data.get_batch(val_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
	    
    x1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    x2 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    logits = models.AlexNet(x1, x2, N_CLASSES)
    loss = tools.loss(logits, y_)  
    acc = tools.accuracy(logits, y_)
    train_op = tools.optimize(loss, LEARNING_RATE)
    
    top_k_op = tf.nn.in_top_k(logits, y_, 1)
             
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
                
                tra_RGB_images,tra_FLOW_images, tra_labels = sess.run([train_RGB_batch, 
                                                                       train_FLOW_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x1:tra_RGB_images, x2:tra_FLOW_images, y_:tra_labels})
                
                if step % 50 == 0:
                    print('Step %d, train loss = %.4f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc))
                    summary_str = sess.run(summary_op, feed_dict={x1:tra_RGB_images, x2:tra_FLOW_images, y_:tra_labels})
                    train_writer.add_summary(summary_str, step)
                    
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_RGB_images, val_FLOW_images, val_labels = sess.run([val_RGB_batch, val_FLOW_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc], 
                                                 feed_dict={x1:val_RGB_images, x2: val_FLOW_images, y_:val_labels})
                    
                    print('**  Step %d, val loss = %.4f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))
                    summary_str = sess.run(summary_op, feed_dict={x1:val_RGB_images, x2: val_FLOW_images, y_:val_labels})
                    val_writer.add_summary(summary_str, step)    
                                    
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
            print('----------------')
            print('Testing Now!')
            print('There are %d test examples'%(n_val))
            
            num_iter = int(math.ceil(n_val / BATCH_SIZE))
            true_count = 0
            total_sample_count = num_iter * BATCH_SIZE
            step = 0
            
            while step < num_iter:
                if coord.should_stop():
                    break
                val_RGB_images, val_FLOW_images, val_labels = sess.run([val_RGB_batch, val_FLOW_batch, val_label_batch])
                predictions = sess.run([top_k_op], feed_dict={x1:val_RGB_images, x2: val_FLOW_images, y_:val_labels})
                true_count += np.sum(predictions)
                step += 1
            precision = true_count / total_sample_count*100.0
            print('precision = %.2f%%' % precision)   
            
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)
        
        
if __name__ == '__main__':
    train_and_test()
