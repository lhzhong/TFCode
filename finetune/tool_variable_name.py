#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:24:40 2018

@author: zhong
"""

import tensorflow as tf

#%%

# First conv
with tf.variable_scope('conv1'):
    w = tf.get_variable(name='weights',
                        shape=[3,3,3,16],
                        initializer=tf.contrib.layers.xavier_initializer(), trainable=False)    
    b = tf.get_variable(name='biases',
                        shape=[16],
                        initializer=tf.constant_initializer(0.0), trainable=False)
    
    print('w name: ', w.name)
    print('b name:', b.name)

with tf.name_scope('conv1'):
    w = tf.get_variable(name='weights',
                        shape=[3,3,3,16],
                        initializer=tf.contrib.layers.xavier_initializer())   
    b = tf.get_variable(name='biases',
                        shape=[16],
                        initializer=tf.constant_initializer(0.0))
    
    print('w name: ', w.name)
    print('b name:', b.name)