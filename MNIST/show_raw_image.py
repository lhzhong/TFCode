#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:03:51 2018

@author: zhong
"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)              # they has been normalized to range (0,1)
#print(mnist.train.images.shape)     # (55000, 28 * 28)
#print(mnist.train.labels.shape)   # (55000, 10)

# plot one example
#plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
#plt.title('%i' % np.argmax(mnist.train.labels[0]))
#plt.show()

# plot multi examples
plt.figure(figsize=(10, 10))
for i in np.arange(0, 49):
    plt.subplot(7, 7, i + 1)
    plt.axis('off')
    plt.imshow(mnist.train.images[i].reshape((28, 28)), cmap='gray')
    plt.title('%i' % np.argmax(mnist.train.labels[i]))
plt.show()
