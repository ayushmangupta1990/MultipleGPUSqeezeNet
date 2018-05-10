#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:12:39 2018

@author: vishalp
"""

import tensorflow as tf
from tensorflow.python.lib.io import file_io
import cv2
import numpy as np
import random
import glob
from multiprocessing import cpu_count

def input_batch_fn(npy_file, epochs, batch_size, num_gpus):
    filenames = np.load(file_io.FileIO(npy_file, 'r'))
    def input_fn():
        Dataset = tf.data.Dataset.from_tensor_slices(filenames)
        Dataset = Dataset.shuffle(buffer_size = 1280)
        Dataset = Dataset.prefetch(2048)
        Dataset = Dataset.map(input_parser_2, num_parallel_calls = cpu_count())
        Dataset = Dataset.apply(tf.contrib.data.ignore_errors())
        Dataset = Dataset.repeat(epochs)
        Dataset = Dataset.prefetch(8)
        Dataset = Dataset.batch(batch_size)
        iterator = Dataset.make_one_shot_iterator()
        feats, labs = iterator.get_next()
        return feats, labs


    def aggregate_batches_npy():
        features = []
        labels = []
        # add if GPU exists condition here to fit GPU and CPU data processing
        if num_gpus > 0:
            num_devices = num_gpus
        else:
            num_devices = 1
        for i in range(num_devices):
            _features, _labels = input_fn()
            features.append(_features)
            labels.append(_labels)
        return features, labels
    return aggregate_batches_npy
    
def input_batch_fn_csv(csv_file, epochs, batch_size, num_gpus):
    def input_csv_fn():
        #filenames = np.load(file_io.FileIO(npy_file, 'r'))
        Dataset = tf.data.TextLineDataset(csv_file).skip(1).shuffle(buffer_size = 2000000).map(parser_csv, num_parallel_calls = cpu_count())
        print("in input fn Dataset 1")
        #Dataset = Dataset.prefetch(2560)
        #Dataset = Dataset.shuffle(buffer_size = 1280)
        Dataset = Dataset.map(input_parser, num_parallel_calls = cpu_count())
        Dataset = Dataset.apply(tf.contrib.data.ignore_errors())
        Dataset = Dataset.repeat(epochs)
        Dataset = Dataset.batch(batch_size)
        Dataset = Dataset.prefetch(batch_size)
        iterator = Dataset.make_one_shot_iterator()
        feats, labs = iterator.get_next()
        print("return feats and labs")
        return feats, labs
        
    def aggregate_csv_batches():
        features = []
        labels = []
        # add if GPU exists condition here to fit GPU and CPU data processing
        if num_gpus > 0:
            num_devices = num_gpus
        else:
            num_devices = 1
        for i in range(num_devices):
            _features, _labels = input_csv_fn()
            features.append(_features)
            labels.append(_labels)
            print("feature and Labels")
        return features, labels
    print("final data Return __")
    return aggregate_csv_batches

def input_batch_fn_folder(data_dir,epochs, batch_size, num_gpus):
    def input_batch_fn(sub_dir):
        filenames = glob.glob(data_dir + sub_dir + '/*.jpg')
        Dataset = tf.data.Dataset.from_tensor_slices(filenames)
        Dataset = Dataset.map(input_parser_2, num_parallel_calls=cpu_count())
        Dataset = Dataset.repeat(epochs)
        Dataset = Dataset.batch(batch_size)
        iterator = Dataset.make_one_shot_iterator()
        feats, labs = iterator.get_next()
        return feats, labs
    
    def aggregate_batches():
        
        features = []
        labels = []
        if num_gpus > 0:
            num_devices = num_gpus
        else:
            num_devices = 1
        for i in range(num_devices):
            feat0, lab0 = input_batch_fn('Ankle')
            feat1, lab1 = input_batch_fn('Crop')
            feat2, lab2 = input_batch_fn('Full')
            _features = tf.concat([feat0,feat1,feat2], axis = 0) 
            _labels = tf.concat([lab0, lab1, lab2], axis = 0)
            features.append(_features)
            labels.append(_labels)
        return features, labels
    return aggregate_batches


def parser_csv(line):
    parsed_line = tf.decode_csv(line, [['string'], ['tf.int64'],['tf.int64']])
    label = parsed_line[-1]
    augment = parsed_line[1]
    return parsed_line[0], augment, label

    
def preprocess_cv(image, augment, label):
    if augment == '1':
        #print('Augmenting the image')
        theta = random.choice([5,10,15,20,-5,-10.-15,-20])
        height, width = image.shape[:2]
        M = cv2.getRotationMatrix2D(((width)/2, (height)/2), theta, 1)
        image = cv2.warpAffine(image, M, (width, height))
        image = cv2.resize(image, (256,256))/255.
        #print("imgaug")
    else:
        #print('No Augmentation')
        image = cv2.resize(image, (256,256))/255.
    return image, label

def preprocess_cv_2(filename, image):
    if filename.split('/')[4] == 'nothing':
        label = 0
        image = cv2.resize(image, (256,256))/255.
    elif filename.split('/')[4] == 'smart':
        label = 1
        image = augment(image)
    else:
        label = 2
        image = augment(image)
    return image, label

def augment(image):
    theta = random.choice([90,180,270])
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D(((width)/2, (height)/2), theta, 1)
    image = cv2.warpAffine(image, M, (width, height))
    image = cv2.resize(image, (256,256))/255.
    return image

def input_parser_plain(filename, augment, label):
    binary = tf.read_file(filename)
    image = tf.image.decode_jpeg(binary, channels = 3)
    image = tf.image.resize_images(image, [256,256])
    image = tf.divide(image, 255.)
    label = tf.string_to_number(label, tf.int32)
    label = tf.one_hot(label,2)
    return image, label


def input_parser(filename, augment, label):
    binary = tf.read_file(filename)
    image = tf.image.decode_image(binary, channels = 3)
    inputs = tf.py_func(preprocess_cv,[image, augment, label], [tf.double,'string'])
    inputs[0] = tf.cast(inputs[0], tf.float32)
    inputs[1] = tf.string_to_number(inputs[1], tf.int32)
    inputs[1] = tf.one_hot(inputs[1], 3)
    return inputs[0], inputs[1]

def input_parser_2(filename):
    binary = tf.read_file(filename)
    image = tf.image.decode_image(binary, channels = 3)
    inputs = tf.py_func(preprocess_cv_2,[filename, image],[tf.double, tf.int64])
    inputs[0] = tf.cast(inputs[0], tf.float32)
    inputs[1] = tf.one_hot(inputs[1], 2)
    return inputs[0], inputs[1]



'''
my_inputs = input_fn(epochs = 2, batch_size = 8)

with tf.Session() as sess:
    while True:
        try:
            elem = sess.run(my_inputs)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
'''
