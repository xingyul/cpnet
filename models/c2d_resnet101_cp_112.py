import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tensorflow as tf
import numpy as np
import tf_util
import copy
import net_utils

def name_mapping(var_dict, debug=False):
    keys = var_dict.keys()
    mapped_dict = {}
    for k in keys:
        key = k.split(':0')[0]
        new_key = key
        if '/W' in key:
            new_key = key.replace('/W', '/weights')
        elif '/mean/EMA' in key:
            new_key = key.replace('/mean/EMA', '/moving_mean')
        elif '/variance/EMA' in key:
            new_key = key.replace('/variance/EMA', '/moving_variance')
        mapped_dict[new_key] = var_dict[k]
    if debug:
        mapped_dict['fc/biases'] = var_dict['linear/b:0']
        mapped_dict['fc/weights'] = var_dict['linear/W:0']
    return mapped_dict

def convert_2d_3d(var_dict):
    keys = var_dict.keys()
    converted_dict = copy.deepcopy(var_dict)
    for k in keys:
        if 'weights' in k and 'conv' in k:
            W = var_dict[k]
            if len(W.shape) == 4:
                W = np.expand_dims(W, 0)
                var = [v for v in tf.global_variables() if k in v.op.name][0]
                if var.get_shape()[0].value != W.shape[0]:
                    W = np.tile(W, [3,1,1,1,1]) / 3
            converted_dict[k] = W
        if 'fc/weights' in k:
            W = var_dict[k]
            converted_dict[k] = W
    return converted_dict

def placeholder_inputs(batch_size, num_frames, height, width, evaluate=False):
    video_pl = tf.placeholder(tf.float32, shape=(batch_size, num_frames, height, width, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return video_pl, labels_pl

def get_model(video, is_training, num_classes=400, bn_decay=0.999, weight_decay=0.0001, pool_t=False, pool_first=False, freeze_bn=False):
    """ Video Net, input is BxTxHxWx3, output Bx400 """
    bsize = video.get_shape()[0].value
    end_points = {}

    channel_stride = [(64, 1), (128, 2), (256, 2), (512, 2)]
    # res block options
    num_blocks = [3, 4, 23, 3]
    # pointnet options
    topks = [None, 4, 4, None]
    shrink_ratios = [None, 2, None, None]
    pn = [[0]*3, [0,0,1,1], [0]*20+[1]*3, [0]*3]

    net = tf_util.conv3d(video, 64, [1, 7, 7], stride=[1, 2 if pool_first else 1, 2 if pool_first else 1], bn=True, bn_decay=bn_decay, is_training=is_training, weight_decay=weight_decay, freeze_bn=freeze_bn, scope='conv0')
    net = tf_util.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='pool0', padding='SAME')

    for gp, cs in enumerate(channel_stride):
        n_channels = cs[0]
        stride = cs[1]
        with tf.variable_scope('group{}'.format(gp)):
            for i in range(num_blocks[gp]):
                with tf.variable_scope('block{}'.format(i)):
                    end_points['res{}_{}_in'.format(gp, i)] = net
                    if i == 0:
                        net_bra = tf_util.conv3d(net, n_channels, [1, 1, 1], stride=[1, stride, stride], bn=True, bn_decay=bn_decay, \
                                is_training=is_training, weight_decay=weight_decay, freeze_bn=freeze_bn, scope='conv1')
                    else:
                        net_bra = tf_util.conv3d(net, n_channels, [1, 1, 1], stride=[1, 1, 1], bn=True, bn_decay=bn_decay, \
                                is_training=is_training, weight_decay=weight_decay, freeze_bn=freeze_bn, scope='conv1')
                    net_bra = tf_util.conv3d(net_bra, n_channels, [1, 3, 3], stride=[1, 1, 1], bn=True, bn_decay=bn_decay, \
                            is_training=is_training, weight_decay=weight_decay, freeze_bn=freeze_bn, scope='conv2')
                    net_bra = tf_util.conv3d(net_bra, n_channels * 4, [1, 1, 1], stride=[1, 1, 1], bn=True, bn_decay=bn_decay, \
                            is_training=is_training, activation_fn=None, weight_decay=weight_decay, freeze_bn=freeze_bn, scope='conv3')
                    if net.get_shape()[-1].value != (n_channels * 4):
                        net = tf_util.conv3d(net, n_channels * 4, [1, 1, 1], stride=[1, stride, stride], bn=True, bn_decay=bn_decay, \
                                is_training=is_training, activation_fn=None, weight_decay=weight_decay, freeze_bn=freeze_bn, scope='convshortcut')
                    net = net + net_bra
                    end_points['res{}_{}_mid'.format(gp, i)] = net
                    if pn[gp][i]:
                        c = net.get_shape()[-1].value
                        net_pointnet, end_point = net_utils.cp_module(net, k=topks[gp], mlp0=[c//8], mlp=[c//8], scope='pointnet', is_training=is_training, bn_decay=bn_decay, \
                                weight_decay=weight_decay, distance='l2', activation_fn=None, freeze_bn=freeze_bn, shrink_ratio=shrink_ratios[gp])
                        net += net_pointnet
                        end_points['pointnet{}_{}'.format(gp, i)] = end_point
                        end_points['after_pointnet{}_{}'.format(gp, i)] = net
                    net = tf.nn.relu(net)
                    end_points['res{}_{}_out'.format(gp, i)] = net

    net = tf.reduce_mean(net, [1,2,3])
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp')
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, weight_decay=weight_decay, scope='fc')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,8,240,320,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
