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
import tools
import os

N_CLASSES = 6
IMG_W = 120
IMG_H = 80
BATCH_SIZE = 128
CAPACITY = 2000
data_dir = './data/KTH_RGB/'
model_dir = './model/KTH_RGB/'
val_txt = 'val.txt'

if not os.path.exists(val_txt):
    print('no data')


def draw_confusion_matrix(show_confusion_matrix=True):
  
    with tf.Graph().as_default():
        
        predictions_label = []
        true_label = []

        val_batch, val_label_batch, n_test = input_data.get_batch(val_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        logits = models.AlexNet(x, N_CLASSES)
        y_pred = tf.nn.softmax(logits)
        y_pred_cls = tf.argmax(y_pred, axis=1)
    
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
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            try:
                
                num_iter = int(math.ceil(n_test / BATCH_SIZE))

                step = 0
    
                while step < num_iter and not coord.should_stop():
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    true_label = np.append(true_label, val_labels)
                    predictions = sess.run([y_pred_cls], feed_dict={x: val_images})
                    predictions = np.array(predictions)
                    predictions = np.squeeze(predictions)
                    predictions_label = np.append(predictions_label, predictions)
                    
                    step += 1
                predictions_label = np.int32(predictions_label)
                true_label = np.int32(true_label)
                # Plot the confusion matrix, if desired.
                if show_confusion_matrix:
                    print("Confusion Matrix:")
                    tools.plot_confusion_matrix(label_true=true_label, label_pred=predictions_label, num_classes=N_CLASSES)
                
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    draw_confusion_matrix(show_confusion_matrix=True)
