#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:15:27 2018

@author: zhong
"""


import os
import os.path

import numpy as np
import tensorflow as tf

import cifar10_input
import model

BATCH_SIZE = 128
learning_rate = 1e-3
MAX_STEP = 10000 # with this setting, it took less than 30 mins on my laptop to train.

def train_running():
    
    data_dir = './data/'
    log_dir = './logs10000/'
    model_dir = './model/'
    
    images, labels = cifar10_input.read_cifar10(data_dir=data_dir,
                                                is_train=True,
                                                batch_size= BATCH_SIZE,
                                                shuffle=True)
    logits = model.inference(images, BATCH_SIZE)
    loss = model.losses(logits, labels)
    accuarcy = model.accuracy(logits, labels)
    train_op = model.optimizer(loss, learning_rate)
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, loss_value, accuarcy_value = sess.run([train_op, loss, accuarcy])
               
            if step % 50 == 0:                 
                print ('Step: %d, loss: %.4f' % (step, loss_value))
                
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)                
    
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