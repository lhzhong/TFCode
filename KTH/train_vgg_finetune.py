#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:25:37 2018

@author: zhong
"""
    
import os
import os.path
import numpy as np
import tensorflow as tf

import input_data
import models
import tools

N_CLASSES = 6
IMG_W = 120  
IMG_H = 120
RATIO = 0.2 
BATCH_SIZE = 128
CAPACITY = 2000
MAX_STEP = 60000
learning_rate = 1e-3

pre_trained_weights = './vgg16_pretrain/vgg16.npy'
data_dir = './data/KTH_RGB/'
model_dir = './model/KTH_RGB/'
logs_train_dir = './logs/KTH_RGB_train/'
logs_val_dir = './logs/KTH_RGB_val/'

def train_running():
    
    with tf.name_scope('input'):
        train_image, train_label, val_image, val_label = input_data.get_files(data_dir, RATIO)
        train_batch, train_label_batch = input_data.get_batch(train_image,
                                                      train_label,
                                                      IMG_W,
                                                      IMG_H,
                                                      BATCH_SIZE, 
                                                      CAPACITY)
        val_batch, val_label_batch = input_data.get_batch(val_image,
                                                      val_label,
                                                      IMG_W,
                                                      IMG_H,
                                                      BATCH_SIZE, 
                                                      CAPACITY)
        
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    logits = models.VGG16N(x, N_CLASSES)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    train_op = tools.optimize(loss, learning_rate)   
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()   
       
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    # load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])   


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tra_summary_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
                
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],feed_dict={x:tra_images, y_:tra_labels})   
            
            if step % 50 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_:tra_labels})
                tra_summary_writer.add_summary(summary_str, step)
                
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],feed_dict={x:val_images,y_:val_labels})
                print('**  Step %d, val loss = %.4f, val accuracy = %.4f%%  **' %(step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={x:val_images,y_:val_labels})
                val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    train_running()
