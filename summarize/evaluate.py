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
IMG_W = 80
IMG_H = 60
BATCH_SIZE = 128
CAPACITY = 2000
val_txt = 'val.txt'
model_dir = './model/KTH_RGB/'


def evaluate_running():
  
    with tf.Graph().as_default():
        
        val_batch, val_label_batch, n_test = input_data.get_batch(val_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        
        model = models.model(x, N_CLASSES)  
        model.AlexNet()
        logits = model.fc3
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
                
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
    
                while step < num_iter and not coord.should_stop():
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    predictions = sess.run([top_k_op], feed_dict={x: val_images, y_: val_labels})
                    true_count += np.sum(predictions)
                    step += 1
                precision = true_count / total_sample_count*100.0
                print('precision = %.2f%%' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    evaluate_running()
