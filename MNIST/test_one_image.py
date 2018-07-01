#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:38:20 2018

@author: zhong
"""

#Evaluate one image
# when training, comment the following codes.

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import models 

num_classes = 10                                                     #### notice
model_dir = './lenet5_model/'
img_dir = '0.jpeg'

with tf.Graph().as_default():
    
    image = tf.gfile.FastGFile(img_dir, 'rb').read()
    image = tf.image.decode_jpeg(image)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image, [28, 28], method=0)
    image = tf.image.per_image_standardization(image)
    image_reshape = tf.reshape(image, [1, 28, 28, 1])

    model = models.Model(image_reshape, num_classes)
    model.lenet5()
    logits = model.logits
    
    with tf.Session() as sess:
        
            
        image_show = sess.run(image)
        
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            
        logits_pro= tf.nn.softmax(logits)
        prediction = sess.run(logits_pro)
        max_index = np.argmax(prediction)
        
        print(prediction)
        
        plt.imshow(image_show[:,:,0], cmap='gray')
        predict_label = "Predict label: {0}".format(max_index)
        plt.title(predict_label)
        plt.show()

                