#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:40:58 2018

@author: zhong
"""

import tensorflow as tf
import tools

class myConfig(object):
    """RNN配置参数"""

    # 模型参数
    num_classes = 6        # 类别数
#    timestep_size = 1   # 时序持续长度为120，即每做一次预测，需要输入120行
#    input_size = 128      # 每个时刻的输入特征是80维的，就是每个时刻输入一行，一行有 80 个像素
    num_layers= 2           # 隐藏层层数
    hidden_dim = 256        # 隐藏层神经元
    rnn = 'lstm'             # lstm 或 gru
    learning_rate = 1e-3    # 学习率
    img_w = 80
    img_h = 60

    
class myModel(object):
    """RNN模型"""
    
    def __init__(self, config):
        
        self.config = config
        # 待输入的数据
        self.input_x = tf.placeholder(tf.float32, [128, self.config.img_w, self.config.img_h, 1], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [128], name='input_y')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size') 
        self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
        self.is_pretrain = True
        self.cnn_rnn()

    def cnn_rnn(self):
        """rnn模型"""
        
        """cnn模型"""
        with tf.name_scope('cnn'):
            self.conv1 = tools.conv('conv1', self.input_x , 16, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.is_pretrain)
            self.pool1 = tools.pool('pool1', self.conv1, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], is_max_pool=True, is_norm=True)

            self.conv2 = tools.conv('conv2', self.pool1, 16, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=self.is_pretrain)
            self.pool2 = tools.pool('pool2', self.conv2, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], is_max_pool=True, is_norm=True)

            self.fc1 = tools.FC_layer('local3', self.pool2, out_nodes=128)
            self.norm1 = tools.batch_norm('batch_norm1', self.fc1)

            self.fc2 = tools.FC_layer('local4', self.norm1, out_nodes=128)
            self.norm2 = tools.batch_norm('batch_norm2', self.fc2)

            # self.fc3 = tools.FC_layer('softmax_linear', self.norm2, out_nodes=self.n_classes

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.name_scope("rnn"):
            _batch, _fcsize = self.norm2.get_shape().as_list()
            X = tf.reshape(self.norm2, [_batch, 1, _fcsize])
            timestep_size = 1
#            input_size = _fcsize
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            init_state = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs = list()
            state = init_state
            for timestep in range(timestep_size):
                (cell_output, state) = rnn_cell(X[:, timestep, :], state)
                outputs.append(cell_output)
            h_state = outputs[-1]
            
        with tf.name_scope("score"):
            # LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
            # 全连接层，后面接dropout以及relu激活
            W = tf.Variable(tf.truncated_normal([self.config.hidden_dim, self.config.num_classes], stddev=0.1), dtype=tf.float32)
            bias = tf.Variable(tf.constant(0.1,shape=[self.config.num_classes]), dtype=tf.float32)
            self.logits = tf.nn.bias_add(tf.matmul(h_state, W), bias)
              
        with tf.name_scope("loss") as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y, name='cross-entropy')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope+'loss', self.loss)
            
        with tf.name_scope("accuracy") as scope:
            correct = tf.nn.in_top_k(self.logits, self.input_y, 1)
            correct = tf.cast(correct, tf.float32)
            self.acc = tf.reduce_mean(correct)*100.0
            tf.summary.scalar(scope+'accuracy', self.acc)     
            
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(self.loss)