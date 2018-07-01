#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:15:27 2018

@author: zhong
"""

import os
import os.path
import numpy as np
import tensorflow as tf
import math

import cifar10_input
import tools

from models.lenet5 import LeNet5
from models.alexnet import AlexNet
from models.vgg16 import VGG16

num_classes = 10
batch_size = 128
learning_rate = 1e-4
dropout_rate = 0.5
max_step = 10000 
img_h = 32
img_w = 32

data_path = './data/'
filewriter_path = './tensorboard/vgg16/'
checkpoint_path = './checkpoints/vgg16/'


def evaluate(sess, val_image, val_label):
    
    val_num = 10000
    val_num_iter = int(math.ceil(val_num / batch_size))
    total_loss = 0.0
    total_acc = 0.0

    for _ in np.arange(val_num_iter):
        val_images, val_labels = sess.run([val_image, val_label])
        val_loss, val_acc = sess.run([loss, acc],
                                     feed_dict={xs: val_images, ys: val_labels, keep_prob: 1.0})
        total_loss += val_loss
        total_acc += val_acc

    return total_loss/val_num_iter, total_acc/val_num_iter


def train_running():

    with tf.Session() as sess:
        
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(filewriter_path, sess.graph)
        
        try:
            for step in np.arange(max_step):
                if coord.should_stop():
                        break
                
                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={xs: tra_images, ys: tra_labels, keep_prob: dropout_rate})
                   
                if step % 50 == 0:                 
                    print ('Step %d, train loss = %.4f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op, 
                                           feed_dict={xs: tra_images, ys: tra_labels, keep_prob: dropout_rate})
                    summary_writer.add_summary(summary_str, step)

                if step % 200 == 0 or (step + 1) == max_step:
                    val_loss, val_acc = evaluate(sess, val_batch, val_label_batch)
                    print('     **val loss = %.4f, val accuracy = %.2f%%**' % (val_loss, val_acc*100.0))
        
                if step % 2000 == 0 or (step + 1) == max_step:
                    checkpoint = os.path.join(checkpoint_path, 'model.ckpt')
                    saver.save(sess, checkpoint, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            
        coord.join(threads)


if __name__ == '__main__':
    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            train_batch, train_label_batch = cifar10_input.read_cifar10(data_path=data_path,
                                                                        is_train=True,
                                                                        batch_size= batch_size,
                                                                        shuffle=True)
            val_batch, val_label_batch = cifar10_input.read_cifar10(data_path=data_path,
                                                                    is_train=False,
                                                                    batch_size= batch_size,
                                                                    shuffle=False)
        
        xs = tf.placeholder(tf.float32, shape=[batch_size, img_h, img_w, 3])
        ys = tf.placeholder(tf.int32, shape=[batch_size, num_classes])
        keep_prob = tf.placeholder(tf.float32)
        
        model = VGG16(xs, num_classes, keep_prob)
        logits = model.logits

        loss = tools.losses(logits, ys)
        acc = tools.accuracy(logits, ys)
        train_op = tools.optimizer(loss, learning_rate)
        train_running()