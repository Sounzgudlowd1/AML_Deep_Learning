# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:18:40 2018

@author: Erik
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 0.0001
epochs = 10
batch_size = 50

x = tf.placeholder(tf.float32, [None, 784])

x_shaped = tf.reshape(x, [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, 10])