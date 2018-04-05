#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:03:51 2018

@author: zhong
"""



import os
import numpy as np
import tensorflow as tf
import input_data
import model



N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
RATIO = 0.2 # take 20% of dataset as validation data 
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 1000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001

   
# you need to change the directories to yours.
train_dir = './data/'
model_dir = './model/'
logs_train_dir = './logs/train/'
logs_val_dir = '.logs/val/'



train, train_label, val, val_label = input_data.get_files(train_dir, RATIO)
train_batch, train_label_batch = input_data.get_batch(train,
                                              train_label,
                                              IMG_W,
                                              IMG_H,
                                              BATCH_SIZE, 
                                              CAPACITY)
val_batch, val_label_batch = input_data.get_batch(val,
                                              val_label,
                                              IMG_W,
                                              IMG_H,
                                              BATCH_SIZE, 
                                              CAPACITY)



x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

logits = model.inference(x, BATCH_SIZE, N_CLASSES)
loss = model.losses(logits, y_)  
acc = model.evaluation(logits, y_)
train_op = model.trainning(loss, learning_rate)


         
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess= sess, coord=coord)
#    summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope))
    summary_op = tf.summary.merge_all()        
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, acc], feed_dict={x:tra_images, y_:tra_labels})
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_:tra_labels})
                train_writer.add_summary(summary_str, step)
#                
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc], feed_dict={x:val_images, y_:val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                summary_str = sess.run(summary_op, feed_dict={x:val_images, y_:val_labels})
                val_writer.add_summary(summary_str, step)  
#                                
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()           
    coord.join(threads)
