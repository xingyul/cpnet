""" Video PointNet Layers
Author: Xingyu Liu
Date: Jul 2018
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, ''))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/knn_l2'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/knn_dot'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/knn_cos'))
import tf_grouping
import tensorflow as tf
import numpy as np
import tf_util

def knn_l2(net, k, u):
    '''
    Input:
        k: int32, number of k in k-nn search
        net: (batch_size, npoint, c) float32 array, points
        u: int32, block size
    Output:
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    INF = 1e8
    batch_size = net.get_shape()[0].value
    npoint = net.get_shape()[1].value
    n_channel = net.get_shape()[2].value

    square = tf.square(tf.norm(net, axis=-1, keepdims=True),)
    def u_block(batch_size, npoint, u):
        block = np.zeros([batch_size, npoint, npoint])
        n = npoint // u
        for i in range(n):
            block[:, (i*u):(i*u+u), (i*u):(i*u+u)] = np.ones([batch_size, u, u]) * (-INF)
        return block
    minus_distance = 2 * tf.matmul(net, tf.transpose(net, [0, 2, 1])) - square - tf.transpose(square, [0, 2, 1]) + tf.constant(u_block(batch_size, npoint, u), dtype=tf.float32)
    _, indices = tf.nn.top_k(minus_distance, k=k, sorted=False)
    return indices

def get_coord(video):
    batch_size = video.get_shape()[0].value
    num_frames = video.get_shape()[1].value
    height = video.get_shape()[2].value
    width = video.get_shape()[3].value

    grid_f, grid_h, grid_w = tf.meshgrid(   tf.range(num_frames, dtype=tf.float32) / num_frames,
                                            tf.range(height, dtype=tf.float32) / height,
                                            tf.range(width, dtype=tf.float32) / width, indexing='ij')
    grid = tf.stack([grid_f, grid_h, grid_w], axis=-1)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1, 1, 1])
    grid = tf.reshape(grid, [batch_size, -1, 3])
    return grid

def cp_module(video, k, mlp, scope, mlp0=None, is_training=None, bn_decay=None, weight_decay=None, data_format='NHWC', distance='l2', activation_fn=None, shrink_ratio=None, freeze_bn=False):
    """
    Args:
        video: (batch_size, T, H, W, C) TF tensor
        k: int -- Top k closest points
        mlp: list of int32 -- output size for MLP on each point
    Returns:
        new features: (batch_size, num_points, mlp[-1]) TF tensor
    """
    batch_size = video.get_shape()[0].value
    num_frames = video.get_shape()[1].value
    height = video.get_shape()[2].value
    width = video.get_shape()[3].value
    num_channels = video.get_shape()[-1].value
    end_points = {}

    if shrink_ratio is not None:
        new_height = height // shrink_ratio
        new_width = width // shrink_ratio
    else:
        new_height = height
        new_width = width

    end_points['input_video'] = video

    if mlp0 is not None:
        with tf.variable_scope(scope) as sc:
            for i, num_out_channel in enumerate(mlp0):
                video = tf_util.conv3d(video, num_out_channel, [1,1,1], padding='VALID',
                         stride=[1,1,1], bn=True, is_training=is_training,
                         scope='conv_bottle%d'%(i), bn_decay=bn_decay, weight_decay=weight_decay, activation_fn=None, freeze_bn=freeze_bn)

    num_channels_bottleneck = video.get_shape()[-1].value
    if shrink_ratio is not None:
        video = tf.reshape(video, [-1, height, width, num_channels_bottleneck])
        video = tf.image.resize_images(video, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)

    net = tf.reshape(video, [batch_size, -1, num_channels_bottleneck])

    if distance == 'l2':
        import knn_l2 as knn
    elif distance == 'dot':
        import knn_dot as knn
    elif distance == 'cos':
        import knn_cos as knn
    nn_idx = knn.knn(net, k, new_height * new_width)

    net_expand = tf.tile(tf.expand_dims(net, axis=2), [1,1,k,1])
    net_grouped = tf_grouping.group_point(net, nn_idx)

    coord = get_coord(tf.reshape(video, [batch_size, -1, new_height, new_width, num_channels_bottleneck]))
    coord_expand = tf.tile(tf.expand_dims(coord, axis=2), [1,1,k,1])
    coord_grouped = tf_grouping.group_point(coord, nn_idx)
    coord_diff = coord_grouped - coord_expand

    end_points['coord'] = {'coord': coord, 'coord_grouped': coord_grouped, 'coord_diff': coord_diff}

    net = tf.concat([coord_diff, net_expand, net_grouped], axis=-1)

    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_out_channel, [1,1], padding='VALID',
                    stride=[1,1], bn=True, is_training=is_training,
                    scope='conv%d'%(i), bn_decay=bn_decay, weight_decay=weight_decay,
                    data_format=data_format, freeze_bn=freeze_bn)
    end_points['before_max'] = net
    net = tf.reduce_max(net, axis=[2], keepdims=True, name='maxpool')
    end_points['after_max'] = net
    net = tf.reshape(net, [batch_size, num_frames, new_height, new_width, mlp[-1]])

    with tf.variable_scope(scope) as sc:
        net = tf_util.conv3d(net, num_channels, [1, 1, 1], stride=[1, 1, 1], bn=False, activation_fn=None, weight_decay=weight_decay, scope='conv_final')
        net = tf.contrib.layers.batch_norm(net, center=True, scale=True,
                is_training=is_training if not freeze_bn else tf.constant(False, shape=(), dtype=tf.bool), decay=bn_decay, updates_collections=None,
                scope='bn_final', data_format=data_format, param_initializers={'gamma': tf.constant_initializer(0., dtype=tf.float32)}, trainable=not freeze_bn)

    if shrink_ratio is not None:
        net = tf.reshape(net, [-1, new_height, new_width, num_channels])
        net = tf.image.resize_images(net, [height, width], method=tf.image.ResizeMethod.BILINEAR)
        net = tf.reshape(net, [batch_size, -1, height, width, num_channels])

    if activation_fn is not None:
        net = activation_fn(net)

    return net, end_points


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sys.path.append(os.path.join(ROOT_DIR, '..'))
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            points = tf.placeholder(dtype=tf.float32, shape=[2,4,14,14,64])
            points_np = np.random.randn(2,4,14,14,64).astype('float32')
            k = 12
            mlp = [32, 64, 128]
            scope = 'test'
            output = cp_module(points, k, mlp, scope, is_training=tf.constant(True, dtype=tf.bool, shape=()), bn_decay=0.9)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        output_np, points_np = sess.run([output, points], feed_dict={points: points_np})

    print(output_np[0,0,0,0])
    print(points_np[0,0,0,0])


