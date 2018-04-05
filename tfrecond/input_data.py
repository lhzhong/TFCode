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
import skimage.io as io

#%%

def get_files(file_dir, ratio, ret_val_num=False):
    '''
    Args:
        file_dir: file directory
        ratio: the ration of val samples
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
def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#%%

def convert_to_tfrecord(images, labels, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
    
    filename = os.path.join(name + '.tfrecords')
    n_samples = len(labels)
    
    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform %s data......'%name)
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i]) 
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                            'label':int64_feature(label),
                            'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
    
    
#%%    
def generate_tfrecond(data_dir):

    name_train = 'train'
    name_val = 'val'
    tra_images,tra_labels,val_images,val_labels = get_files(data_dir, 0.2)
    convert_to_tfrecord(tra_images, tra_labels, name_train)
    convert_to_tfrecord(val_images, val_labels, name_val)

#%%  
def read_and_decode(tfrecords_file, image_W, image_H, batch_size, min_after_dequeue):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        image_W: image width
        image_H: image height
        batch_size: number of images in each batch
        capacity: the maximum elements in queue
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [160,120,3])
    label = tf.cast(img_features['label'], tf.int32)  
    
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H) 
    image = tf.image.per_image_standardization(image)
    capacity = 2*min_after_dequeue
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size= batch_size,
                                                min_after_dequeue= min_after_dequeue, 
                                                capacity = capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch

#%%  
# To test the generated batches of images

#import matplotlib.pyplot as plt
#import numpy as np
#
#BATCH_SIZE = 4
#min_after_dequeue = 512
#IMG_W = 160
#IMG_H = 120
#
##train_dir = './data/KTH_RGB_small/'
#tfrecords_file ='train.tfrecords'
#ratio = 0.2
##tra_images, tra_labels, val_images, val_labels = input_data.get_files(train_dir, ratio)
#tra_image_batch, tra_label_batch = read_and_decode(tfrecords_file, IMG_W, IMG_H, BATCH_SIZE, min_after_dequeue)
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