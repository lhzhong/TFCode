#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:36:24 2018

@author: zhong
"""

import tensorflow as tf
import os
import numpy as np
import input_data
from cnn_lstm_model import myConfig,myModel

CAPACITY = 2000
TRAIN_RATIO = 0.7
BATCH_SIZE = 128
MAX_STEP = 10000

data_dir = './data/KTH_RGB/'
model_dir = './model/'
new_model_dir = './model/new/'
logs_train_dir = './logs/train/'
logs_val_dir = './logs/val/'
train_txt = 'train.txt'
val_txt = 'val.txt'

if not os.path.exists(train_txt):
    input_data.generate_txt(data_dir, TRAIN_RATIO)

if not os.path.exists('./model'):
    os.mkdir('./model')

def retrain_running():
    
    with tf.variable_scope('input'):
        train_batch, train_label_batch, _ = input_data.get_batch(train_txt, model.config.img_w, model.config.img_h, BATCH_SIZE, CAPACITY)
        val_batch, val_label_batch, _ = input_data.get_batch(val_txt, model.config.img_w, model.config.img_h, BATCH_SIZE, CAPACITY)    
    
    with tf.Session() as sess:
        
        saver = tf.train.Saver(tf.global_variables())
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
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                tra_loss, tra_acc,  _ = sess.run([model.loss, model.acc, model.optim], 
                                     feed_dict={model.input_x: tra_images, model.input_y: tra_labels, 
                                                model.keep_prob: 0.5, model.batch_size: BATCH_SIZE})

                if step % 50 == 0:
                    print('Step %d, train loss = %.4f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc))
                    summary_str = sess.run(summary_op, 
                                           feed_dict={model.input_x: tra_images, model.input_y: tra_labels, 
                                                      model.keep_prob: 0.5, model.batch_size: BATCH_SIZE})
                    train_writer.add_summary(summary_str, step)
                #
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([model.loss, model.acc],
                                                 feed_dict={model.input_x: val_images, model.input_y: val_labels, 
                                                            model.keep_prob: 1.0, model.batch_size: BATCH_SIZE})

                    print('**  Step %d, val loss = %.4f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))
                    summary_str = sess.run(summary_op,
                                           feed_dict={model.input_x: val_images, model.input_y: val_labels, 
                                                      model.keep_prob: 1.0, model.batch_size: BATCH_SIZE})
                    val_writer.add_summary(summary_str, step)
                    #
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(new_model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
 
if __name__ == '__main__':
    config = myConfig()
    model = myModel(config)
    retrain_running()
