#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:43:13 2018

@author: zhong
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#%%
def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers. 
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)) # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.1))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

#%%
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True, is_norm = False):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    with tf.variable_scope(layer_name):
        
        if is_max_pool:
            x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
        else:
            x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
            
        if is_norm == True:
            x = tf.nn.lrn(x, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm')
        
        return x

#%%
def batch_norm(layer_name,x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    with tf.variable_scope(layer_name):
        epsilon = 1e-3
        batch_mean, batch_var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x,
                                      mean=batch_mean,
                                      variance=batch_var,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=epsilon)
        return x

#%%
def FC_layer(layer_name, x, out_nodes):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.1))
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x

#%%
def loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss
    
#%%
def accuracy(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
  """
  with tf.name_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float32)
      accuracy = tf.reduce_mean(correct)*100.0
#      correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
#      correct = tf.cast(correct, tf.float32)
#      accuracy = tf.reduce_mean(correct)*100.0
      tf.summary.scalar(scope+'/accuracy', accuracy)
      return accuracy


#%%
def optimize(loss, learning_rate):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False) 
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

#%%
def plot_feature_map(feature_map):
    
    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flat):
        feature_map = np.array(feature_map)
        feature_map = np.squeeze(feature_map)
        img = feature_map[0,:,:,i]
#        ax.imshow(img, interpolation='nearest', cmap='seismic')
        ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
        
#%%
def print_all_variables(train_only=True):
    if train_only:
        t_vars = tf.trainable_variables()
        print("Trainable variables:------------------------")
    else:
        t_vars = tf.global_variables()
        print("Global variables:------------------------")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))   

#%%
