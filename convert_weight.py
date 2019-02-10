# coding: utf-8
from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parse_anchors, load_weights

num_class = 5
img_size = 512
weight_path = './data/darknet_weights/yolov3.weights'
save_path = './data/darknet_weights/yolov3.ckpt'
anchors = parse_anchors('./data/yolo_anchors.txt')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.48)
model = yolov3(num_class, anchors)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.variable_scope('yolov3'):
        feature_map = model.forward(inputs)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

    load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))
