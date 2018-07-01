#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:17:46 2018

@author: zhong
"""

import tensorflow as tf
import tools


class Model(object):
    
    def __init__(self, input_data, n_classes, keep_prob=1.0, is_trainable=True):
        
        self.input = input_data
        self.n_classes = n_classes
        self.keep_prob = keep_prob
        self.is_trainable = is_trainable
        
    def lenet_300_100(self):
    
        with tf.name_scope('LeNet_300_100'):
            
            self.fc1 = tools.fc_layer('fc1', self.input, out_nodes=300)
            self.fc2 = tools.fc_layer('fc2', self.fc1, out_nodes=100)
            self.dropout1 = tools.dropout('dropout1', self.fc1, self.keep_prob)
            self.logits = tools.fc_layer('fc3', self.dropout1, use_relu=False, out_nodes=self.n_classes)

    def lenet5(self):
    
        with tf.name_scope('LeNet5'):
            
            self.conv1 = tools.conv('conv1', self.input, 32, kernel_size=[5,5], stride=[1,1,1,1], is_trainable=self.is_trainable)
            self.pool1 = tools.pool('pool1', self.conv1, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
            self.conv2 = tools.conv('conv2', self.pool1, 64, kernel_size=[5,5], stride=[1,1,1,1], is_trainable=self.is_trainable)
            self.pool2 = tools.pool('pool2', self.conv2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
            self.fc1 = tools.fc_layer('fc1', self.pool2, out_nodes=512)
            self.dropout1 = tools.dropout('dropout1', self.fc1, self.keep_prob)
    
            self.logits = tools.fc_layer('fc2', self.dropout1, use_relu=False, out_nodes=self.n_classes)