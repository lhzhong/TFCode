#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:17:46 2018

@author: zhong
"""

import tensorflow as tf
import tools


class model(object):
    
    def __init__(self, input_data, n_classes, is_pretrain=True):
        
        self.input = input_data
        self.n_classes = n_classes
        self.is_pretrain = is_pretrain

    def LeNet(self):
    
        with tf.name_scope('LeNet'):
            
            self.conv1 = tools.conv('conv1', self.input, 32, kernel_size=[5,5], stride=[1,1,1,1], is_trainable=is_trainable)
            self.pool1 = tools.pool('pool1', self.conv1, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
            self.conv2 = tools.conv('conv2', self.pool1, 64, kernel_size=[5,5], stride=[1,1,1,1], is_trainable=is_trainable)
            self.pool2 = tools.pool('pool2', self.conv2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
            self.full_connection = tools.FC_layer('full_connection', self.pool2, out_nodes=128)
            self.output = tools.FC_layer('output', self.full_connection, out_nodes=n_classes,use_relu=False)

    def AlexNet(self):
        
        with tf.name_scope('AlexNet'):
            
            self.conv1 = tools.conv('conv1', self.input, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.pool1 = tools.pool('pool1', self.conv1, kernel=[1,3,3,1], stride=[1,2,2,1], is_max_pool=True, is_norm=True)
            
            self.conv2 = tools.conv('conv2', self.pool1, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.pool2 = tools.pool('pool2', self.conv2, kernel=[1,3,3,1], stride=[1,2,2,1], is_max_pool=True, is_norm=True)
            
            self.fc1 = tools.FC_layer('local3', self.pool2, out_nodes=128)
            self.norm1 = tools.batch_norm('batch_norm1', self.fc1)
    
            self.fc2 = tools.FC_layer('local4', self.norm1, out_nodes=128)
            self.norm2 = tools.batch_norm('batch_norm2', self.fc2)
    
            self.fc3 = tools.FC_layer('softmax_linear', self.norm2, out_nodes=self.n_classes)

    def VGG16(self):
    
        with tf.name_scope('VGG16'):
    
            self.conv1_1 = tools.conv('conv1_1', self.input, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)   
            self.conv1_2 = tools.conv('conv1_2', self.conv1_1, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)   
            self.pool1 = tools.pool('pool1', self.conv1_2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
                
            self.conv2_1 = tools.conv('conv2_1', self.pool1, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)    
            self.conv2_2 = tools.conv('conv2_2', self.conv2_1, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.pool2 = tools.pool('pool2', self.conv2_2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
             
            self.conv3_1 = tools.conv('conv3_1', self.pool2, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.conv3_2 = tools.conv('conv3_2', self.conv3_1, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.conv3_3 = tools.conv('conv3_3', self.conv3_2, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.pool3 = tools.pool('pool3', self.conv3_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
                
            self.conv4_1 = tools.conv('conv4_1', self.pool3, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.conv4_2 = tools.conv('conv4_2', self.conv4_1, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.conv4_3 = tools.conv('conv4_3', self.conv4_2, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.pool4 = tools.pool('pool4', self.conv4_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        
            self.conv5_1 = tools.conv('conv5_1', self.pool4, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.conv5_2 = tools.conv('conv5_2', self.conv5_1, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.conv5_3 = tools.conv('conv5_3', self.conv5_2, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=self.is_pretrain)
            self.pool5 = tools.pool('pool5', self.conv5_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)             
            
            self.fc6 = tools.FC_layer('fc6', self.pool5, out_nodes=4096)        
            self.batch_norm1 = tools.batch_norm('batch_norm1', self.fc6)         
            
            self.fc7 = tools.FC_layer('fc7', self.batch_norm1, out_nodes=4096)        
            self.batch_norm2 = tools.batch_norm('batch_norm2', self.fc7)  
              
            self.fc8 = tools.FC_layer('fc8', self.batch_norm2, out_nodes=self.n_classes)