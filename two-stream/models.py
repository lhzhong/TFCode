#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:01:54 2018

@author: zhong
"""

import tensorflow as tf
import tools

#%%
def AlexNet(x, y,n_classes, is_pretrain=True):
    
    with tf.name_scope('rgb-AlexNet'):
        
        x = tools.conv('conv1-rgb', x, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.pool('pool1-rgb', x, kernel=[1,3,3,1], stride=[1,2,2,1], is_max_pool=True, is_norm=True)
        
        x = tools.conv('conv2-rgb', x, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.pool('pool2-rgb', x, kernel=[1,3,3,1], stride=[1,2,2,1], is_max_pool=True, is_norm=True) 
        
        x = tools.FC_layer('local3-rgb', x, out_nodes=128)
        
    with tf.name_scope('flow-AlexNet'):
        
        y = tools.conv('conv1-flow', y, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        y = tools.pool('pool1-flow', y, kernel=[1,3,3,1], stride=[1,2,2,1], is_max_pool=True, is_norm=True)
        
        y = tools.conv('conv2-flow', y, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        y = tools.pool('pool2-flow', y, kernel=[1,3,3,1], stride=[1,2,2,1], is_max_pool=True, is_norm=True)
        
        y = tools.FC_layer('local3-flow', y, out_nodes=128)
    
    with tf.name_scope('fusion'):    
        
        z = x + y
            
        z = tools.FC_layer('local4', z, out_nodes=128)
        z = tools.FC_layer('softmax_linear', z, out_nodes=n_classes)
        
    return z
#%%     
