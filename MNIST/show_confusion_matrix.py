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

num_classes = 10
model_dir =  './lenet5_model/'

def show_confusion_matrix(show_example_errors=False, show_confusion_matrix=False):
    
    with tf.Graph().as_default():

        mnist = input_data.read_data_sets('../MNIST_data/',one_hot=True)
 
        x = tf.placeholder(tf.float32, shape=[None, 784])
        x_reshape = tf.reshape(x, [-1, 28,28,1])
        y_ = tf.placeholder(tf.float32,[None, num_classes])
        
        model = models.Model(x_reshape, num_classes)
        model.lenet5()
        logits = model.logits
        
        y_pred_one = tf.argmax(logits,1)
        y_true_one = tf.argmax(y_,1)
    
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
                
            
            label_pred = sess.run(y_pred_one, feed_dict={x: mnist.test.images}) 
            label_true = sess.run(y_true_one, feed_dict={y_: mnist.test.labels})
            
            label_true = np.argmax(mnist.test.labels, axis=1)
            correct = (label_true == label_pred)
            correct_sum = correct.sum()
            num_test = len(mnist.test.images)
            acc = float(correct_sum) / num_test
            
            # Print the accuracy.
            msg = "Accuracy on Test-Set: {0:.2%} ({1} / {2})"
            print(msg.format(acc, correct_sum, num_test))
        
            # Plot some examples of mis-classifications, if desired.
            if show_example_errors:
                print("Example errors:")
                
                correct = (label_true == label_pred)
                tools.plot_example_errors(mnist, label_pred=label_pred, correct=correct)
        
            # Plot the confusion matrix, if desired.
            if show_confusion_matrix:
                print("Confusion Matrix:")
                tools.plot_confusion_matrix(label_true=label_true, label_pred=label_pred, num_classes=num_classes)
        
if __name__ == '__main__':
    show_confusion_matrix(show_example_errors=True,show_confusion_matrix=True)