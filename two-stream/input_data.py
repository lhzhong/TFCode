#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:03:17 2018

@author: zhong
"""


import tensorflow as tf
import numpy as np
import os
import math


def getPerActionList(actionPath, isRGB=True):

    actionList = []
    actionNames = os.listdir(actionPath)
    if len(actionNames) > 0:
        for action in actionNames:
            actionName = os.path.join(actionPath, action)
            x = os.listdir(actionName)
            if len(x) > 0:
                if isRGB:
                    for xt in x[:-1]:
                        fullfilename = os.path.join(actionName, xt)
                        actionList.append(fullfilename)
                else:
                    for xt in x:
                        fullfilename = os.path.join(actionName, xt)
                        actionList.append(fullfilename)
    return actionList


def getAllActionList(actionRGBPath, actionFLOWPath, actionNum):

    img_label_list = []
    class_num = 0
    for i in actionNum:
        imageRGBList = getPerActionList(actionRGBPath + i, True)
        imageFLOWList = getPerActionList(actionFLOWPath + i, True)
        for img_RGB, img_FLow in zip(imageRGBList, imageFLOWList):
            _img_label = img_RGB + ' ' + img_FLow + ' ' + str(class_num)
            img_label_list.append(_img_label)
        class_num = class_num + 1
    np.random.shuffle(img_label_list)
    return img_label_list


def get_txt(TXTFILE, IMAGEFILE):

    txtFile = open(TXTFILE, 'w')
    num = 0
    for i in IMAGEFILE:
        t = i + '\n'
        txtFile.writelines(t)
        num = num + 1
    txtFile.close()
    return num


def split_txt(file):
    img_label = []
    with open(file, 'r') as file_to_read:
        for line in file_to_read.readlines():
            line = line.strip('\n')
            tmp = line.split(" ")
            img_label.append(tmp)
        img_label = np.array(img_label)
        img_RGB = img_label[:, 0]
        img_Flow = img_label[:, 1]
        label = img_label[:, -1]
        label = np.int32(label)
    return img_RGB, img_Flow, label


def generate_txt(data_dir_one, data_dir_two, ratio):

    trainFile = './train.txt'
    testFile = './val.txt'

    ACTIONNUM = os.listdir(data_dir_one)

    img_label = getAllActionList(data_dir_one, data_dir_two, ACTIONNUM)

    train_img_label = img_label[:int(len(img_label) * ratio)]
    test_img_label = img_label[int(len(img_label) * ratio):]
    train_num = get_txt(trainFile, train_img_label)
    test_num = get_txt(testFile, test_img_label)

    print('Successfully generated txt file')
    print('Trainging num: %d' % train_num)
    print('Testing num: %d' % test_num)

def get_batch(file, image_W, image_H, batch_size, capacity):

    img_label = []
    with open(file, 'r') as file_to_read:
        for line in file_to_read.readlines():
            line = line.strip('\n')
            tmp = line.split(" ")
            img_label.append(tmp)
        img_label = np.array(img_label)
        img_RGB = img_label[:, 0]
        img_Flow = img_label[:, 1]
        label = img_label[:, -1]
        label = np.int32(label)
        

    num = len(img_label)
    img_RGB = tf.cast(img_RGB, tf.string)
    img_Flow = tf.cast(img_Flow, tf.string)
    label = tf.cast(label, tf.int64)

    # make an input queue
    input_queue = tf.train.slice_input_producer([img_RGB, img_Flow, label])
    
    label = input_queue[2]
    image_RGB_contents = tf.read_file(input_queue[0])
    image_FLOW_contents = tf.read_file(input_queue[1])
    image_RGB = tf.image.decode_jpeg(image_RGB_contents, channels=3)
    image_FLOW = tf.image.decode_jpeg(image_FLOW_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################

    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image_RGB = tf.image.resize_images(image_RGB, [image_W, image_H], method=0)
    image_FLOW = tf.image.resize_images(image_FLOW, [image_W, image_H], method=0)

    image_RGB = tf.image.per_image_standardization(image_RGB)
    image_FLOW = tf.image.per_image_standardization(image_FLOW)

    image_RGB_batch, image_FLOW_batch, label_batch = tf.train.batch([image_RGB, image_FLOW, label],
                                                                    batch_size=batch_size,
                                                                    num_threads=64,
                                                                    capacity=capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_RGB_batch = tf.cast(image_RGB_batch, tf.float32)
    image_FLOW_batch = tf.cast(image_FLOW_batch, tf.float32)
    
    return image_RGB_batch, image_FLOW_batch, label_batch, num