#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:30:03 2018

@author: zhong
"""

import os

train_data_dir = '/path/to/train_data/'
val_data_dir = '/path/to/train_data/'
test_data_dir = '/path/to/train_data/'


def get_data_path(dataset="train"):
    if dataset == "train":
        data_dir = train_data_dir
    elif dataset == "val":
        data_dir = val_data_dir
    else:
        data_dir = test_data_dir

    txt_file = '%s.txt' % dataset
    class_num = 0
    img_label_list = []

    categorys = os.listdir(data_dir)
    for category in categorys:
        # Get all files in current category's folder.
        folder_path = os.path.join(data_dir, category)  # e.g. './KTH/boxing/'
        filenames = sorted(os.listdir(folder_path))

        for filename in filenames:

            filepath = os.path.join(folder_path, filename)
            imagenames = sorted(os.listdir(filepath))

            for image in imagenames:
                imagepath = os.path.join(filepath, image)
                _img_label = imagepath + ' ' + str(class_num)
                img_label_list.append(_img_label)
        class_num += 1

    # np.random.shuffle(img_label_list)
    write_txt(txt_file, img_label_list)


def write_txt(txt_file, image):

    txt = open(txt_file, 'w')
    num = 0
    for i in image:
        t = i + '\n'
        txt.writelines(t)
        num = num + 1
    txt.close()


if __name__ == "__main__":
    print("Making train dataset txt file.")
    get_data_path(dataset="train")
    print("Making raw val dataset")
    get_data_path(dataset="val")
    print("Making raw test dataset")
    get_data_path(dataset="test")
