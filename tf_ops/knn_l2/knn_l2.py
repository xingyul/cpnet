#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import ops

package_path = os.path.dirname(os.path.realpath(__file__))
knn_module = tf.load_op_library(os.path.join(package_path, 'knn_op.so'))

def knn(I, k, u):
    '''
    I has shape of [B, N, C]
    '''
    I_norm = tf.square(tf.norm(I, axis=-1))
    k = tf.ones([k], dtype=tf.int32)
    u = tf.ones([u], dtype=tf.int32)
    return knn_module.knn_l2(I, I_norm, k, u)

@ops.RegisterShape('KnnL2')
def _my_knn_shape(op):
    shape1 = op.inputs[0].get_shape().with_rank(3)
    B = shape1.dims[0]
    N = shape1.dims[1]
    shape2 = op.inputs[2].get_shape().with_rank(1)
    K = shape2.dims[0]
    return [tf.TensorShape([B, N, K])]

