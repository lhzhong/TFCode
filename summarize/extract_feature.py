#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:50:31 2018

@author: zhong
"""

import numpy as np
import os
import tensorflow as tf
import input_data
import models
import tools
import math
import scipy.io as sio 

N_CLASSES = 6
IMG_W = 80
IMG_H = 60
BATCH_SIZE = 128
CAPACITY = 2000
train_txt = 'train.txt'
val_txt = 'val.txt'
model_dir = './model/KTH_RGB/'

if not os.path.exists('./features'):
    os.mkdir('./features')


def fc_feature_extract(txtfile, istrainFile=True):
  
    with tf.Graph().as_default():

        batch, label_batch, n = input_data.get_batch(txtfile, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        
        model = models.model(x, N_CLASSES)  
        model.AlexNet()
        feature = model.fc2
    
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            try:
                
                num_iter = int(math.ceil(n / BATCH_SIZE))
                step = 0    
                
                while step < num_iter and not coord.should_stop():
                    
                    images, labels = sess.run([batch, label_batch])    
                    
                    if step == 0 :
                        all_features = sess.run([feature],feed_dict={x:images})
                        all_features = np.array(all_features)
                        all_features = np.squeeze(all_features)
                        
                        all_labels = labels
                        pass
                    
                    feature_map = sess.run([feature],feed_dict={x:images})
                    feature_map = np.array(feature_map)
                    feature_map = np.squeeze(feature_map)
#                    dic_feature = {'features':feature_map}
#                    sio.savemat('./features/'+str(step)+'.mat',dic_feature)
                    all_features = np.concatenate((all_features, feature_map),axis=0)
                    all_labels = np.concatenate((all_labels, labels),axis=0)

                    step = step+1
                
                if istrainFile==True:
                    dic_feature = {'features': all_features}
                    sio.savemat('./features/'+'train_features.mat', dic_feature)
                    dic_label = {'labels': all_labels}
                    sio.savemat('./features/'+'train_labels.mat', dic_label)
                    print('\nSuccessfully generated train matfile\n')
                else:
                    dic_feature = {'features': all_features}
                    sio.savemat('./features/' + 'val_features.mat', dic_feature)
                    dic_label = {'labels': all_labels}
                    sio.savemat('./features/' + 'val_labels.mat', dic_label)
                    print('\nSuccessfully generated val matfile\n')
                        
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
            coord.join(threads)


def conv_feature_extract(txtfile):
  
    with tf.Graph().as_default():

        batch, label_batch, n = input_data.get_batch(txtfile, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        
        model = models.model(x, N_CLASSES)  
        model.AlexNet()
        feature = model.conv1
    
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            try:
                
                num_iter = int(math.ceil(n / BATCH_SIZE))
                step = 0    

                while step < num_iter and not coord.should_stop():
                    
                    images, labels = sess.run([batch, label_batch]) 
                    
                    feature_map = sess.run([feature],feed_dict={x: images})
                    feature_map = np.array(feature_map)
                    feature_map = np.squeeze(feature_map)
                    
                    if step ==0:
                        tools.print_all_variables()
                    
                    print("batch %d-0: feature map------------------------"%step)
                    tools.plot_feature_map(feature_map)

                    step = step+1
   
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    # conv_feature_extract(train_txt)
    fc_feature_extract(train_txt, istrainFile=True)
    fc_feature_extract(val_txt, istrainFile=False)