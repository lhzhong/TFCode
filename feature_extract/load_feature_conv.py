#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:40:08 2018

@author: zhong
"""

import scipy.io as sio 
import numpy as np
from  sklearn import svm
from sklearn import metrics
from datetime import datetime 

start_time = datetime.now()
train_features = sio.loadmat('./features/train_features.mat')
train_labels = sio.loadmat('./features/train_labels.mat')
val_features = sio.loadmat('./features/val_features.mat')
val_labels = sio.loadmat('./features/val_labels.mat')

train_feature = np.array(train_features['train_features'][0:400])
train_feature_flatter = train_feature.reshape([400, -1])
train_label = np.array(train_labels['train_labels'])
train_label_flatter = train_label.ravel()[0:400]

val_feature = np.array(val_features['val_features'][0:200])
val_feature_flatter = val_feature.reshape([200, -1])
val_label = np.array(val_labels['val_labels'])
val_label_flatter = val_label.ravel()[0:200]

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(train_feature_flatter, train_label_flatter)

val_predict = clf.predict(val_feature_flatter)
accuarcy = metrics.accuracy_score(val_predict, val_label_flatter)
# accuarcy =clf.score(x_test,y_test)
print('Accuarcy: %f' % (accuarcy*100.0))

end_time = datetime.now()
print((end_time-start_time))