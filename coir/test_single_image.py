# coding: utf-8
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time

from modelcoir import yolov3
import sys
sys.path[:0] = ['../']
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image_ir", type=str, default = '../data/ir512/00001270_ir.png', help="The path of the ir input image.")
parser.add_argument("--input_image_co", type=str, default = '../data/visible512/00001270_co.png', help="The path of the co input image.")
parser.add_argument("--anchor_path", type=str, default="../data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[512, 512],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="../data/vedai.names",
                    help="The path of the class names.")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

img_co = cv2.imread(args.input_image_co)
img_ori = cv2.imread(args.input_image_ir)
height_ori, width_ori = img_ori.shape[:2]
img_ir = cv2.resize(img_ori, tuple(args.new_size))
img_co = cv2.resize(img_co, tuple(args.new_size))
img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)
img_co = cv2.cvtColor(img_co, cv2.COLOR_BGR2RGB)
img_ir = np.asarray(img_ir, np.float32)
img_co = np.asarray(img_co, np.float32)
img_ir = img_ir[np.newaxis, :] / 255.
img_co = img_co[np.newaxis, :] / 255.

restore_path = tf.train.latest_checkpoint(args.checkpoint_dir)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    input_data_ir = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data_ir')
    input_data_co = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data_co')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data_ir, input_data_co, False)

    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, iou_thresh=0.5)

    saver = tf.train.Saver()
    saver.restore(sess, restore_path)

    start = time.time()
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data_ir: img_ir,
                                                                            input_data_co: img_co})
    print(time.time()-start)

    # rescale the coordinates to the original image
    boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
    boxes_[:, 3] *= (height_ori/float(args.new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(10000)
