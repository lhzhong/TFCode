#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:17:46 2018

@author: zhong
"""

import tensorflow as tf
import numpy as np


class VGG16(object):

    def __init__(self, input_data, num_classes, keep_prob=1.0, skip_layer=[],
                 weights_path='DEFAULT'):
        self.input = input_data
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.skip_layer = skip_layer
        
        if weights_path == 'DEFAULT':
            self.weights_path = './pretrain/vgg16.npy'
        else:
            self.weights_path = weights_path
            
        self.creat()

    def creat(self):
        
        self.conv1_1 = conv(self.input, 3, 3, 64, 1, 1, name='conv1_1')
        self.conv1_2 = conv(self.conv1_1, 3, 3, 64, 1, 1, name='conv1_2')
        self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, name='pool1')
        
        self.conv2_1 = conv(self.pool1, 3, 3, 128, 1, 1, name='conv2_1')
        self.conv2_2 = conv(self.conv2_1, 3, 3, 128, 1, 1, name='conv2_2')
        self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, name='pool2')
        
        self.conv3_1 = conv(self.pool2, 3, 3, 256, 1, 1, name='conv3_1')
        self.conv3_2 = conv(self.conv3_1, 3, 3, 256, 1, 1, name='conv3_2')
        self.conv3_3 = conv(self.conv3_2, 3, 3, 256, 1, 1, name='conv3_3')
        self.pool3 = max_pool(self.conv3_3, 2, 2, 2, 2, name='pool3')
        
        self.conv4_1 = conv(self.pool3, 3, 3, 512, 1, 1, name='conv4_1')
        self.conv4_2 = conv(self.conv4_1, 3, 3, 512, 1, 1, name='conv4_2')
        self.conv4_3 = conv(self.conv4_2, 3, 3, 512, 1, 1, name='conv4_3')
        self.pool4 = max_pool(self.conv4_3, 2, 2, 2, 2, name='pool4')
    
        self.conv5_1 = conv(self.pool4, 3, 3, 512, 1, 1, name='conv5_1')
        self.conv5_2 = conv(self.conv5_1, 3, 3, 512, 1, 1, name='conv5_2')
        self.conv5_3 = conv(self.conv5_2, 3, 3, 512, 1, 1, name='conv5_3')
        self.pool5 = max_pool(self.conv5_3, 2, 2, 2, 2, name='pool5')            
    
        self.fc6 = fc(self.pool5, 4096, name='fc6')
        #x = tools.batch_norm(x)
        self.fc7 = fc(self.fc6, 4096, name='fc7')
        #x = tools.batch_norm(x)
        self.logits = fc(self.fc7, self.num_classes, relu=False, name='fc8')
    
    def load_initial_weights(self, session):
        """Load weights from file into network.
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """

        weights_dict = np.load(self.weights_path, encoding='latin1').item()

        for op_name in weights_dict:
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))
        print('Sucessful load weights!')


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[filter_height,
                                         filter_width,
                                         input_channels / groups,
                                         num_filters],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[num_filters],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))

    if groups == 1:
        act = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        act = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(act, biases), tf.shape(act))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_out, name, relu=True):
    """Create a fully connected layer."""
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights',
                                  shape=[size, num_out],
                                  trainable=True,
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[num_out],
                                 trainable=True,
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(flat_x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,

                                              bias=bias, name=name)


def batch_norm(x):
    """Batch normlization(I didn't include the offset and scale)
    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    return tf.nn.batch_normalization(x,
                                     mean=batch_mean,
                                     variance=batch_var,
                                     offset=None,
                                     scale=None,
                                     variance_epsilon=epsilon)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

