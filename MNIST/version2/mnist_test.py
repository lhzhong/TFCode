#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:02:32 2018

@author: zhong
"""

import tensorflow as tf
import numpy as np
import models
import tools
from tensorflow.examples.tutorials.mnist import input_data

img_size = 28
num_channels = 1
num_classes = 10

def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    
    with tf.Graph().as_default():

        data = input_data.read_data_sets('data/MNIST/', one_hot=True)
        data.test.labels_one = np.argmax(data.test.labels, axis=1)
        
        test_batch_size = 256
        num_test = len(data.test.images)
        label_pred = np.zeros(shape=num_test, dtype=np.int)
        i = 0
        
        x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name='x')
        x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        
        predict = models.LeNet(x_image, num_classes)
        y_pred = tf.nn.softmax(predict)
        y_pred_one = tf.argmax(y_pred,1)
        
        saver = tf.train.Saver(tf.global_variables())
        model_dir = './model/'
    
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
                
            # get batch data    
            while i < num_test:
                j = min(i + test_batch_size, num_test)
                images = data.test.images[i:j, :]
                labels = data.test.labels[i:j, :]
                feed_dict = {x: images,y_true: labels}
                label_pred[i:j] = sess.run(y_pred_one, feed_dict=feed_dict)
                i = j
        
            label_true = np.argmax(data.test.labels, axis=1)
            correct = (label_true == label_pred)
            correct_sum = correct.sum()
            acc = float(correct_sum) / num_test
        
            # Print the accuracy.
            msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
            print(msg.format(acc, correct_sum, num_test))
        
            # Plot some examples of mis-classifications, if desired.
            if show_example_errors:
                print("Example errors:")
                tools.plot_example_errors(data, label_pred=label_pred, correct=correct)
        
            # Plot the confusion matrix, if desired.
            if show_confusion_matrix:
                print("Confusion Matrix:")
                tools.plot_confusion_matrix(data, label_pred=label_pred, num_classes=num_classes)
        
if __name__ == '__main__':
    print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)