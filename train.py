# coding: utf-8
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import logging

from utils.data_utils import parse_data
from utils.misc_utils import parse_anchors, read_class_names, shuffle_and_overwrite, update_dict, make_summary, config_learning_rate, config_optimizer, list_add
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu
from utils.nms_utils import gpu_nms

from model import yolov3

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="YOLO-V3 training procedure.")
# some paths
parser.add_argument("--train_file", type=str, default="./data/annotations/train.txt",
                    help="The path of the training txt file.")

parser.add_argument("--val_file", type=str, default="./data/annotations/val.txt",
                    help="The path of the validation txt file.")

parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")

parser.add_argument("--save_dir", type=str, default="./checkpoint/",
                    help="The directory of the weights to save.")

parser.add_argument("--log_dir", type=str, default="./data/logs/",
                    help="The directory to store the tensorboard log files.")

parser.add_argument("--progress_log_path", type=str, default="./data/logs/progress.log",
                    help="The path to record the training progress.")

parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")

parser.add_argument("--class_name_path", type=str, default="./data/vedai.names",
                    help="The path of the class names.")

# some numbers
parser.add_argument("--batch_size", type=int, default=10,
                    help="The batch size for training.")

parser.add_argument("--img_size", nargs='*', type=int, default=[512, 512],
                    help="Resize the input image to `img_size`, size format: [width, height]")

parser.add_argument("--total_epoches", type=int, default=5000,
                    help="Total epoches to train.")

parser.add_argument("--train_evaluation_freq", type=int, default=100,
                    help="Evaluate on the training batch after some steps.")

parser.add_argument("--val_evaluation_freq", type=int, default=100,
                    help="Evaluate on the whole validation dataset after some steps.")

parser.add_argument("--save_freq", type=int, default=500,
                    help="Save the model after some steps.")

parser.add_argument("--num_threads", type=int, default=1,
                    help="Number of threads for image processing used in tf.data pipeline.")

parser.add_argument("--prefetech_buffer", type=int, default=3,
                    help="Prefetech_buffer used in tf.data pipeline.")

# learning rate and optimizer
parser.add_argument("--optimizer_name", type=str, default='adam',
                    help="The optimizer name. Chosen from [sgd, momentum, adam, rmsprop]")

parser.add_argument("--save_optimizer", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the optimizer parameters into the checkpoint file.")

parser.add_argument("--learning_rate_init", type=float, default=5e-4,
                    help="The initial learning rate.")

parser.add_argument("--lr_type", type=str, default='fixed',
                    help="The learning rate type. Chosen from [fixed, exponential]")

parser.add_argument("--lr_decay_freq", type=int, default=1000,
                    help="The learning rate decay frequency. Used when chosen exponential lr_type.")

parser.add_argument("--lr_decay_factor", type=float, default=0.96,
                    help="The learning rate decay factor. Used when chosen exponential lr_type.")

parser.add_argument("--lr_lower_bound", type=float, default=1e-6,
                    help="The minimum learning rate. Used when chosen exponential lr type.")

# finetune
parser.add_argument("--restore_part", nargs='*', type=str, default=['yolov3/darknet53_body'],
                    help="Partially restore part of the model for finetuning. Set [None] to restore the whole model.")

parser.add_argument("--update_part", nargs='*', type=str, default=['yolov3/yolov3_head'],
                    help="Partially restore part of the model for finetuning. Set [None] to train the whole model.")

# warm up strategy
parser.add_argument("--use_warm_up", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to use warm up strategy.")

parser.add_argument("--warm_up_lr", type=float, default=5e-5,
                    help="Warm up learning rate.")

parser.add_argument("--warm_up_epoch", type=int, default=5,
                    help="Warm up training epoches.")
flag = parser.parse_args()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

# args params
flag.anchors = parse_anchors(flag.anchor_path)
flag.classes = read_class_names(flag.class_name_path)
flag.class_num = len(flag.classes)
flag.train_img_cnt = len(open(flag.train_file, 'r').readlines())
flag.val_img_cnt = len(open(flag.val_file, 'r').readlines())
flag.train_batch_num = int(np.ceil(float(flag.train_img_cnt) / flag.batch_size))
flag.val_batch_num = int(np.ceil(float(flag.val_img_cnt) / flag.batch_size))

# setting loggers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=flag.progress_log_path, filemode='w')

# setting placeholders
is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')

##################
# tf.data pipeline
##################
# Selecting `feedable iterator` to switch between training dataset and validation dataset

# manually shuffle the train txt file because tf.data.shuffle is soooo slow!!
# you can google it for more details.
shuffle_and_overwrite(flag.train_file)
train_dataset = tf.data.TextLineDataset(flag.train_file)
train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(
    lambda x: tf.py_func(parse_data, [x, flag.class_num, flag.img_size, flag.anchors, 'train'], [tf.float32, tf.float32, tf.float32, tf.float32]),
    batch_size=flag.batch_size))
train_dataset = train_dataset.prefetch(flag.prefetech_buffer)

val_dataset = tf.data.TextLineDataset(flag.val_file)
val_dataset = val_dataset.apply(tf.contrib.data.map_and_batch(
    lambda x: tf.py_func(parse_data, [x, flag.class_num, flag.img_size, flag.anchors, 'val'], [tf.float32, tf.float32, tf.float32, tf.float32]),
    batch_size=flag.batch_size))
val_dataset.prefetch(flag.prefetech_buffer)

# creating two dataset iterators
train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

# creating two dataset handles
train_handle = train_iterator.string_handle()
val_handle = val_iterator.string_handle()
# select a specific iterator based on the passed handle
dataset_iterator = tf.data.Iterator.from_string_handle(handle_flag, train_dataset.output_types,
                                                       train_dataset.output_shapes)

# get an element from the choosed dataset iterator
image, y_true_13, y_true_26, y_true_52 = dataset_iterator.get_next()
y_true = [y_true_13, y_true_26, y_true_52]

# tf.data pipeline will lose the data shape, so we need to set it manually
image.set_shape([None, flag.img_size[1], flag.img_size[0], 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################

# define yolo-v3 model here
yolo_model = yolov3(flag.class_num, flag.anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

################
# register the gpu nms operation here for the following evaluation scheme
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, flag.class_num)
################

if flag.restore_part == ['None']:
    flag.restore_part = [None]
if flag.update_part == ['None']:
    flag.update_part = [None]
saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=flag.restore_part))
update_vars = tf.contrib.framework.get_variables_to_restore(include=flag.update_part)

tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
tf.summary.scalar('train_batch_statistics/loss_class', loss[4])

global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
if flag.use_warm_up:
    learning_rate = tf.cond(tf.less(global_step, flag.train_batch_num * flag.warm_up_epoch),
        lambda: flag.warm_up_lr, lambda: config_learning_rate(flag, global_step - flag.train_batch_num * flag.warm_up_epoch))
else:
    learning_rate = config_learning_rate(flag, global_step)
tf.summary.scalar('learning_rate', learning_rate)

if not flag.save_optimizer:
    saver_to_save = tf.train.Saver()

optimizer = config_optimizer(flag.optimizer_name, learning_rate)

if flag.save_optimizer:
    saver_to_save = tf.train.Saver()

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss[0], var_list=update_vars, global_step=global_step)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), train_iterator.initializer])
    train_handle_value, val_handle_value = sess.run([train_handle, val_handle])
    saver_to_restore.restore(sess, flag.restore_path)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(flag.log_dir, sess.graph)

    print('\n----------- start to train -----------\n')

    for epoch in range(flag.total_epoches):
        for i in range(flag.train_batch_num):
            _, summary, y_pred_, y_true_, loss_, global_step_, lr = sess.run([train_op, merged, y_pred, y_true, loss, global_step, learning_rate],
                                                                             feed_dict={is_training: True, handle_flag: train_handle_value})
            writer.add_summary(summary, global_step=global_step_)
            info = "Epoch: {}, global_step: {}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
                epoch, global_step_, loss_[0], loss_[1], loss_[2], loss_[3], loss_[4])
            print(info)
            logging.info(info)

            # evaluation on the training batch
            if global_step_ % flag.train_evaluation_freq == 0 and global_step_ > 0:
                # recall, precision = evaluate_on_cpu(y_pred_, y_true_, flag.class_num, calc_now=True)
                recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred_, y_true_, flag.class_num, calc_now=True)
                info = "===> batch recall: {:.3f}, batch precision: {:.3f} <===".format(recall, precision)
                print(info)
                logging.info(info)

                writer.add_summary(make_summary('evaluation/train_batch_recall', recall), global_step=global_step_)
                writer.add_summary(make_summary('evaluation/train_batch_precision', precision), global_step=global_step_)

            # start to save
            # NOTE: this is just demo. You can set the conditions when to save the weights.
            if global_step_ % flag.save_freq == 0 and global_step_ > 0:
                if loss_[0] <= 5.:
                    saver_to_save.save(sess, flag.save_dir + 'model-step_{}_loss_{:4f}_lr_{:.7g}'.format(global_step_, loss_[0], lr))

            # switch to validation dataset for evaluation
            if global_step_ % flag.val_evaluation_freq == 0 and global_step_ > 0:
                sess.run(val_iterator.initializer)
                true_positive_dict, true_labels_dict, pred_labels_dict = {}, {}, {}
                val_loss = [0., 0., 0., 0., 0.]
                for j in range(flag.val_batch_num):
                    y_pred_, y_true_, loss_ = sess.run([y_pred, y_true, loss],
                                                        feed_dict={is_training: False, handle_flag: val_handle_value})
                    true_positive_dict_tmp, true_labels_dict_tmp, pred_labels_dict_tmp = \
                        evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag,
                                        y_pred_, y_true_, flag.class_num, calc_now=False)
                    true_positive_dict = update_dict(true_positive_dict, true_positive_dict_tmp)
                    true_labels_dict = update_dict(true_labels_dict, true_labels_dict_tmp)
                    pred_labels_dict = update_dict(pred_labels_dict, pred_labels_dict_tmp)

                    val_loss = list_add(val_loss, loss_)

                # make sure there is at least one ground truth object in each image
                # avoid divided by 0
                recall = float(sum(true_positive_dict.values())) / (sum(true_labels_dict.values()) + 1e-6)
                precision = float(sum(true_positive_dict.values())) / (sum(pred_labels_dict.values()) + 1e-6)

                info = "===> Epoch: {}, global_step: {}, recall: {:.3f}, precision: {:.3f}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
                    epoch, global_step_, recall, precision, val_loss[0] / flag.val_img_cnt, val_loss[1] / flag.val_img_cnt, val_loss[2] / flag.val_img_cnt, val_loss[3] / flag.val_img_cnt, val_loss[4] / flag.val_img_cnt)
                print(info)
                logging.info(info)
                writer.add_summary(make_summary('evaluation/val_recall', recall), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_precision', precision), global_step=epoch)

                writer.add_summary(make_summary('validation_statistics/total_loss', val_loss[0] / flag.val_img_cnt), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_xy', val_loss[1] / flag.val_img_cnt), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_wh', val_loss[2] / flag.val_img_cnt), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_conf', val_loss[3] / flag.val_img_cnt), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_class', val_loss[4] / flag.val_img_cnt), global_step=epoch)

        # manually shuffle the training data in a new epoch
        shuffle_and_overwrite(flag.train_file)
        sess.run(train_iterator.initializer)
