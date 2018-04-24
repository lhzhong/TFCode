#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:01:54 2018

@author: zhong
"""

#%%

import tensorflow as tf
import tools

#%%
def AlexNet(x, n_classes, is_pretrain=True):
    
    with tf.name_scope('AlexNet'):
        
        x = tools.conv('conv1', x, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.pool('pool1', x, kernel=[1,3,3,1], stride=[1,2,2,1], is_max_pool=True, is_norm=True)
        
        x = tools.conv('conv2', x, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.pool('pool2', x, kernel=[1,3,3,1], stride=[1,2,2,1], is_max_pool=True, is_norm=True)
        
        x = tools.FC_layer('local3', x, out_nodes=128)
        x = tools.batch_norm('batch_norm1',x)
        
        x = tools.FC_layer('local4', x, out_nodes=128)
        x = tools.batch_norm('batch_norm2', x)
        
        x = tools.FC_layer('softmax_linear', x, out_nodes=n_classes)

        return x
     
#%% 

def VGG16(x, n_classes, is_pretrain=True):
    
    with tf.name_scope('VGG16'):

        x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)   
        x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)   
        x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
        x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)    
        x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
         
        x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
        x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.pool('pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
        x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.pool('pool5', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)             
        
        x = tools.FC_layer('fc6', x, out_nodes=4096)        
        x = tools.batch_norm('batch_norm1', x)         
        
        x = tools.FC_layer('fc7', x, out_nodes=4096)        
        x = tools.batch_norm('batch_norm2', x)  
          
        x = tools.FC_layer('fc8', x, out_nodes=n_classes)
    
        return x

#%%
