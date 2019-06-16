#!/usr/bin/python

import tensorflow as tf
import numpy as np

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('../grouping')
sys.path.append('../../utils')

import tf_grouping
import knn_l2 as knn
import net_utils


B = 2
N = 6
C = 3

np.random.seed(0)


with tf.Graph().as_default():
    with tf.device('/gpu:0'):
        I = tf.placeholder(dtype=tf.float32, shape=[B, N, C])
        k = 2
        u = 2
        o = knn.knn(I, k, u)
        o_tf = net_utils.knn_l2(I, k, u)
        I_g = tf_grouping.group_point(I, o)

    # I_np = np.reshape(np.arange(B*N*C), [B, N, C]) * 0.1
    I_np = np.random.randn(B, N, C)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    print(I_np)
    o_tf_np = sess.run(o, feed_dict={I: I_np})
    print(o_tf_np)
    I_np_sq = np.square(np.linalg.norm(I_np[1], axis=-1, keepdims=True))
    print('')
    print(2 * I_np[1].dot(I_np[1].T) - I_np_sq - np.transpose(I_np_sq, [1, 0]))
    print('')
    print(o_tf_np.shape)

    I_g_np = sess.run(I_g, feed_dict={I: I_np})
    print(I_g_np)
