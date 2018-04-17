#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:03:17 2018

@author: zhong
"""

#%%

import tensorflow as tf
import numpy as np
import os
import math

#%%

def get_files(file_dir, ratio, ret_val_num=False):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    class_train = []
    label_train = []
    k=0
    for train_class in os.listdir(file_dir):
        for sub_train in os.listdir(file_dir+train_class):
            for image in os.listdir(file_dir+train_class+'/'+sub_train) :
                class_train.append(file_dir+train_class+'/'+sub_train+'/'+image)
                label_train.append(k)
        k=k+1
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    #shuffle the samples
    np.random.shuffle(temp)
    #after transpose, images is in dimension 0 and label in dimension 1
    all_image_list = list(temp[:,0])
    all_label_list = list(temp[:,1])
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio)) #测试样本数
    n_train = n_sample - n_val # 训练样本数
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    print("Training picture: %d, %d classes" %(n_train,k))
    print("Testing picture: %d, %d classes" %(n_val,k))
    if ret_val_num:
        return tra_images,tra_labels,val_images,val_labels,n_val
    else:
        return tra_images,tra_labels,val_images,val_labels

#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)    
    # if you want to test the generated batches of images, you might want to comment the following line.
    
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
#%% TEST
# To test the generated batches of images

#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 2
#CAPACITY = 256
#IMG_W = 120
#IMG_H = 120
#
#train_dir = '/home/cpss/zhong/KTH/KTH_RGB/'
#ratio = 0.2
#tra_images, tra_labels, val_images, val_labels = get_files(train_dir, ratio)
#tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([tra_image_batch, tra_label_batch])
#            
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


#%%