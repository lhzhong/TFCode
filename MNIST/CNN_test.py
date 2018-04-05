#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:47:54 2018

@author: zhong
"""

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

def weight_varible(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def convolutional_neural_network(data, keep_prob=0.5):
    
    x_image = tf.reshape(data, [-1,28,28,1])
    
    with tf.variable_scope("conv_layer1") as layer1:
        W_conv1 = weight_varible([5,5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pooling(h_conv1)
    with tf.variable_scope("conv_layer2") as layer2:
        W_conv2 = weight_varible([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pooling(h_conv2)
    
    with tf.variable_scope("full_connection") as full_layer3:
        h_pool2_flat = tf.contrib.layers.flatten(h_pool2)
        nodes = h_pool2_flat.get_shape().as_list()
        W_fc1 = weight_varible([nodes[1], 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    with tf.variable_scope("output") as output_layer4:
        W_fc2 = weight_varible([1024, 10])
        b_fc2 = bias_variable([10])
        predict = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    return predict

def train():

    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32,[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    predict = convolutional_neural_network(xs)
    cross_extropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=tf.argmax(ys, 1))
    cross_extropy_mean = tf.reduce_mean(cross_extropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_extropy_mean)

    saver = tf.train.Saver()
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
    model_dir = './model'
    model_path = model_dir + '/best.ckpt'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("create the directory: %s\n" % model_dir)
    
    with tf.Session() as sess:
        # 若模型数据不存在，需要训练模型参数
        if not os.path.exists(model_path + ".index"):
            sess.run(tf.global_variables_initializer())
            
            for i in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
                if i%50 == 0:
                    y_predict = sess.run(predict, feed_dict={xs:mnist.test.images})
                    correct_predict = tf.equal(tf.argmax(y_predict,1), tf.argmax(mnist.test.labels,1))
                    accuarcy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
                    result = sess.run(accuarcy, feed_dict={xs:mnist.validation.images, ys:mnist.validation.labels, keep_prob:1})
                    print("Validation Accuarcy:%f"%(result))
                    
                    saver.save(sess, model_path, write_meta_graph=False)
                    
        # 恢复数据并校验和测试
        saver.restore(sess, model_path)

        y_predict = sess.run(predict, feed_dict={xs:mnist.test.images})
        correct_predict = tf.equal(tf.argmax(y_predict,1), tf.argmax(mnist.test.labels,1))
        accuarcy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        result = sess.run(accuarcy, feed_dict={xs:mnist.test.images, ys:mnist.test.labels, keep_prob:1})
        print("Test Accuarcy:%f"%(result))
    
if __name__ == '__main__':
    train()    