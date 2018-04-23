#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:34:18 2018

@author: zhong
"""

import numpy as np
import tensorflow as tf
import input_data
import models
import math

N_CLASSES = 6
IMG_W = 120
IMG_H = 120
BATCH_SIZE = 128
CAPACITY = 2000

val_txt = 'val.txt'
model_dir = './model/KTH_twostream/'

def evaluate_running():
  
    with tf.Graph().as_default():
        val_RGB_batch, val_FLOW_batch, val_label_batch, n_val = input_data.get_batch(val_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#        
        x1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        x2 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        logits = models.AlexNet(x1, x2, N_CLASSES)
        top_k_op = tf.nn.in_top_k(logits, y_, 1)
    
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                
                num_iter = int(math.ceil(n_val / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
    
                while step < num_iter and not coord.should_stop():
                    val_RGB_images, val_FLOW_images, val_labels = sess.run([val_RGB_batch, val_FLOW_batch, val_label_batch])
                    predictions = sess.run([top_k_op], feed_dict={x1:val_RGB_images, x2: val_FLOW_images, y_:val_labels})
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.2f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    evaluate_running()