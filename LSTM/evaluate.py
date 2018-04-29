#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 22:26:30 2018

@author: cpss
"""

import tensorflow as tf
import numpy as np
import input_data
import math
from rnn_model import myRNNConfig,myRNN

CAPACITY = 2000
TRAIN_RATIO = 0.7
BATCH_SIZE = 128

data_dir = './data/KTH_RGB/'
model_dir = './model/'
val_txt = 'val.txt'

def evaluate_running():
    with tf.Graph().as_default():
        
        config = myRNNConfig()
        model = myRNN(config)
        
        val_batch, val_label_batch, n_test = input_data.get_batch(val_txt, config.timestep_size, config.input_size, BATCH_SIZE, CAPACITY)
        
        saver = tf.train.Saver(tf.global_variables())
        
        top_k_op = tf.nn.in_top_k(model.logits, model.input_y, 1)
        
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
                    val_loss, val_acc = sess.run([model.loss, model.acc],
                                                 feed_dict={model.input_x: val_images, model.input_y: val_labels, 
                                                            model.keep_prob: 1.0, model.batch_size: BATCH_SIZE})
                    
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    predictions = sess.run([top_k_op],
                                           feed_dict={model.input_x: val_images, model.input_y: val_labels, 
                                                      model.keep_prob: 1.0, model.batch_size: BATCH_SIZE})
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
