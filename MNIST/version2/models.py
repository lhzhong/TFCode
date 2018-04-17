#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:06:14 2018

@author: lcq
"""

import tensorflow as tf
import tools

#%%
def LeNet(x, n_classes, is_trainable=True):
    
    with tf.name_scope('LeNet'):
        
        x = tools.conv('conv1', x, 32, kernel_size=[5,5], stride=[1,1,1,1], is_trainable=is_trainable)
        x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        
        x = tools.conv('conv2', x, 64, kernel_size=[5,5], stride=[1,1,1,1], is_trainable=is_trainable)
        x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        
        x = tools.FC_layer('full_connection', x, out_nodes=128)
        x = tools.FC_layer('output', x, out_nodes=n_classes,use_relu=False)

        return x
