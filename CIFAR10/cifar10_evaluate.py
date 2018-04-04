#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:35:00 2018

@author: zhong
"""

import math
import numpy as np
import tensorflow as tf

import cifar10_input
import model

BATCH_SIZE = 128
learning_rate = 0.05
MAX_STEP = 10000 # with this setting, it took less than 30 mins on my laptop to train.

def evaluate_running():
    with tf.Graph().as_default():
        
        model_dir = './model/'
        test_dir = './data/'
        n_test = 10000
        
        # reading test data
        images, labels = cifar10_input.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size= BATCH_SIZE,
                                                    shuffle=False)

        logits = model.inference(images,BATCH_SIZE)
        labels = tf.cast(labels,tf.int64)
        top_k_op = tf.nn.in_top_k(logits, tf.argmax(labels,1), 1)
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