'''
    Single-GPU training.
'''
import argparse
import math
from datetime import datetime
# import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import cv2
import torch
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import tf_util
import dataloader
from dict_restore import DictRestore
from saver_restore import SaverRestore
import spatial_transforms
import target_transforms
from mean import get_mean, get_std

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1', help='GPU to use [default: GPU 0,1]')
parser.add_argument('--model', default='', help='Model name [default: ]')
parser.add_argument('--model_path', default=None, help='Model snapshot to restore [default: ]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--data', default='', help='Data dir [default: ]')
parser.add_argument('--height', type=int, default=112, help='Video image height [default: 112]')
parser.add_argument('--width', type=int, default=112, help='Video image width [default: 112]')
parser.add_argument('--num_frames', type=int, default=8, help='Number of frames to use [default: 251]')
parser.add_argument('--frame_step', type=int, default=4, help='Frame step [default: 4]')
parser.add_argument('--pool_t', type=int, default=1, help='Whether to pool in time dimension [default: 1]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.002]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: momentum]')
parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay factor [default: 0.0001]')
parser.add_argument('--decay_step', type=int, default=40, help='Decay step (number of epoches) for lr decay [default: 40]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.1]')
parser.add_argument('--num_threads', type=int, default=64, help='Number of threads to use in loading data [default: 64]')
parser.add_argument('--num_classes', type=int, default=400, help='Number of classes [default: 400]')
parser.add_argument('--symmetric_flip_labels', default=None, help='The left-right label pairs [default: None]')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate instead of continue with last training')
parser.add_argument('--freeze_bn', action='store_true', help='Freeze all batch norm layers')
parser.add_argument('--debug', action='store_true', help='Whether to debug load model')
parser.add_argument('--command_file', default=None, help=' [Shell command file to use default: None]')
FLAGS = parser.parse_args()

random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

EPOCH_CNT = 0

NUM_GPUS = len(FLAGS.gpu.split(','))
BATCH_SIZE = FLAGS.batch_size
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE // NUM_GPUS

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
NUM_FRAMES = FLAGS.num_frames
FRAME_STEP = FLAGS.frame_step
POOL_T = FLAGS.pool_t
HEIGHT = FLAGS.height
WIDTH = FLAGS.width
DATA = FLAGS.data
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
WEIGHT_DECAY = FLAGS.weight_decay
EPOCH_DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_THREADS = FLAGS.num_threads
RESET_LR = FLAGS.reset_lr
FREEZE_BN = FLAGS.freeze_bn
DEBUG = FLAGS.debug
SYMMETRIC_FLIP_LABELS = FLAGS.symmetric_flip_labels
COMMAND_FILE = FLAGS.command_file

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
MODEL_PATH = FLAGS.model_path
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
os.system('cp %s %s ' % (COMMAND_FILE, LOG_DIR)) # bkp of command shell file
os.system('cp utils/net_utils.py %s ' % (LOG_DIR)) # bkp of net_utils file
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# train augmentation
normalize = spatial_transforms.ToNormalizedTensor(mean=get_mean(), std=get_std())
train_transform = spatial_transforms.Compose([
    spatial_transforms.RandomResizedCrop(size=(WIDTH, WIDTH), scale=(0.5, 1.0), ratio=(1.- 0.1, 1. + 0.1)),
    # spatial_transforms.RandomHorizontalFlip(),
    spatial_transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    normalize])
# validation transform
val_transform = spatial_transforms.Compose([
    # spatial_transforms.Resize(256),
    spatial_transforms.CenterCrop(WIDTH),
    normalize])
target_transform = target_transforms.ClassLabel()

train_loader, val_loader = dataloader.get_loader(root=DATA, train_transform=train_transform, val_transform=val_transform, target_transform=target_transform,
        batch_size=BATCH_SIZE, num_frames=NUM_FRAMES, step_size=FRAME_STEP, val_samples=1, n_threads=NUM_THREADS)
DECAY_STEP = EPOCH_DECAY_STEP * len(train_loader)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = FLAGS.num_classes

symmetric_flip_labels = {}
if SYMMETRIC_FLIP_LABELS is not None:
    pairs = SYMMETRIC_FLIP_LABELS.split(',')
    for p in pairs:
        p1, p2 = p.split(':')
        symmetric_flip_labels[int(p1)] = int(p2)
        symmetric_flip_labels[int(p2)] = int(p1)

print('symmetric pairs: ', symmetric_flip_labels)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
    Args:
	tower_grads: List of lists of (gradient, variable) tuples. The outer list
	is over individual gradients. The inner list is over the gradient
	calculation for each tower.
    Returns:
	List of pairs of (gradient, variable) where the gradient has been averaged
	across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        if grad_and_vars[0][0] is not None:
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch,               # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            video_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_FRAMES, HEIGHT, WIDTH)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            MODEL.get_model(video_pl, num_classes=NUM_CLASSES if not DEBUG else 1000, is_training=is_training_pl, bn_decay=bn_decay, weight_decay=WEIGHT_DECAY, pool_t=POOL_T, freeze_bn=FREEZE_BN)

            tower_grads = []
            pred_gpu = []
            total_loss_gpu = []
            for i in range(NUM_GPUS):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                        # Evenly split input data to each GPU
                        vd_batch = tf.slice(video_pl,
                            [i*DEVICE_BATCH_SIZE,0,0,0,0], [DEVICE_BATCH_SIZE,-1,-1,-1,-1])
                        label_batch = tf.slice(labels_pl,
                            [i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])

                        pred, end_points = MODEL.get_model(vd_batch, num_classes=NUM_CLASSES if not DEBUG else 1000,
                            is_training=is_training_pl, bn_decay=bn_decay, weight_decay=WEIGHT_DECAY, pool_t=POOL_T, freeze_bn=FREEZE_BN)

                        MODEL.get_loss(pred, label_batch, end_points)
                        losses = tf.get_collection('losses', scope)
                        total_loss = tf.add_n(losses, name='total_loss')
                        for l in losses + [total_loss]:
                            tf.summary.scalar(l.op.name, l)

                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)

                        pred_gpu.append(pred)
                        total_loss_gpu.append(total_loss)

            pred = tf.concat(pred_gpu, 0)
            total_loss = tf.reduce_mean(total_loss_gpu)

            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=batch)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Add ops to save all the variables.
            saver_save = tf.train.Saver(max_to_keep=50)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore variables from disk.
        if MODEL_PATH is not None:
            if 'npz' not in MODEL_PATH:
                sr = SaverRestore(MODEL_PATH, log_string, ignore=['batch:0'] if RESET_LR else [])
                sr.run_init(sess)
                log_string("Model restored.")
            else:
                dict_file = np.load(MODEL_PATH)
                dict_for_restore = {}
                dict_file_keys = dict_file.keys()
                for k in dict_file_keys:
                    dict_for_restore[k] = dict_file[k]
                dict_for_restore = MODEL.name_mapping(dict_for_restore, debug=DEBUG)
                dict_for_restore = MODEL.convert_2d_3d(dict_for_restore)
                dr = DictRestore(dict_for_restore, log_string)
                dr.run_init(sess)
                log_string("npz file restored.")

        if DEBUG:
            im = cv2.imread('green_mamba.jpg').astype('float32')
            if im.shape[0] < im.shape[1]:
                im = cv2.resize(im, (int(256. * float(im.shape[1]) / im.shape[0]), 256))
            else:
                im = cv2.resize(im, (256, int(256. * float(im.shape[0]) / im.shape[1])))
            im = im[int(im.shape[0]/2-112):int(im.shape[0]/2+112), int(im.shape[1]/2-112):int(im.shape[1]/2+112), :]
            im = im / 255
            mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, -1])
            std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, -1])
            im = (im - mean) / std
            wh = WIDTH
            im = cv2.resize(im, (wh, wh))
            im = np.reshape(im, [1, 1, wh, wh, 3])
            im = np.tile(im, (1, NUM_FRAMES, 1, 1, 1))
            pred_np = sess.run(pred, feed_dict={video_pl: im, is_training_pl: False})
            pred_np = np.reshape(pred_np, [-1])
            print(pred_np.argsort()[-5:][::-1])
            exit()



        ops = {'video_pl': video_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            log_string('learning_rate: {}'.format(sess.run(learning_rate)))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, train_loader)

            # Save the variables to disk.
            if epoch % 1 == 0:
                save_path = saver_save.save(sess, os.path.join(LOG_DIR, "model-{}.ckpt".format(epoch)))
                log_string("Model saved in file: %s" % save_path)

            eval_one_epoch(sess, ops, test_writer, val_loader)


def train_one_epoch(sess, ops, train_writer, train_loader):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_data = inputs.data.numpy()
        bsize = batch_data.shape[0]
        batch_label = targets.data.numpy()
        batch_data = np.transpose(batch_data, [0,2,3,4,1])
        if SYMMETRIC_FLIP_LABELS is not None:
            for b in range(bsize):
                if np.random.randint(2) == 1:
                    batch_data[b] = batch_data[b, :, :, ::-1, :]
                    if batch_label[b] in symmetric_flip_labels.keys():
                        batch_label[b] = symmetric_flip_labels[batch_label[b]]

        feed_dict = {ops['video_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        if (batch_idx+1)%10 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0

def eval_one_epoch(sess, ops, test_writer, val_loader):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    total_correct_top1 = 0
    total_correct_top5 = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_data = inputs.data.numpy()
        bsize = batch_data.shape[0]
        batch_label = targets.data.numpy()
        batch_data = np.transpose(batch_data, [0,2,3,4,1])

        feed_dict = {ops['video_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val_top5 = np.argsort(pred_val, 1)[:, ::-1][:, :5]
        pred_val_top1 = np.argmax(pred_val, 1)
        correct_top1 = np.sum(pred_val_top1[0:bsize] == batch_label[0:bsize])
        correct_top5 = np.sum(np.any(pred_val_top5 == np.transpose(np.tile(batch_label[0:bsize], [5, 1])), axis=1))
        total_correct_top1 += correct_top1
        total_correct_top5 += correct_top5
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy top1 : %f'% (total_correct_top1 / float(total_seen)))
    log_string('eval accuracy top5 : %f'% (total_correct_top5 / float(total_seen)))
    EPOCH_CNT += 1


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
