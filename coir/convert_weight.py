# coding: utf-8
from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from modelcoir import yolov3
import sys
sys.path[:0] = ['../']
from utils.misc_utils import parse_anchors, load_weights

img_size = 512
weight_path = '../data/darknet_weights/yolov3.weights'
save_path = './data/darknet_weights/yolov3.ckpt'
anchors = parse_anchors('../data/yolo_anchors.txt')
class_num = 5

model = yolov3(class_num, anchors)
with tf.Session() as sess:
    inputs_ir = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
    inputs_co = tf.placeholder(tf.float32, [1, img_size, img_size, 3])


    with tf.variable_scope('yolov3'):
        feature_map = model.forward(inputs_ir, inputs_co)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

    varlist_dark_ir = tf.global_variables(scope='yolov3/darknet53_body_ir')
    varlist_dark_co = tf.global_variables(scope='yolov3/darknet53_body_co')
    varlist_head = tf.global_variables(scope='yolov3/yolov3_head')
    var_list_dark_head = varlist_dark_ir + varlist_head

    init = tf.global_variables_initializer()
    sess.run(init)
    load_ops = load_weights(varlist_dark_co, weight_path)
    sess.run(load_ops)
    load_ops2 = load_weights(varlist_dark_ir, weight_path)
    sess.run(load_ops2)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))
