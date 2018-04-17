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
IMG_W = 120  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 120
RATIO = 0.2 # take 20% of dataset as validation data 
BATCH_SIZE = 100
CAPACITY = 2000

def evaluate_running():
  
    with tf.Graph().as_default():
        data_dir = './data/KTH_RGB/'
        model_dir = './model/KTH_RGB6000/'
        train_image, train_label, val_image, val_label,n_test = input_data.get_files(data_dir, RATIO, ret_val_num=True)
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
#        
        logits = models.AlexNet(val_batch, N_CLASSES)
        top_k_op = tf.nn.in_top_k(logits, val_label_batch, 1)
    
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
                
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
    
                while step < num_iter and not coord.should_stop():
                    val_images_,val_labels_ = sess.run([val_batch, val_label_batch])
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    evaluate_running()