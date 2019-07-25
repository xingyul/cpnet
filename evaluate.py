'''
    Evaluate classification performance
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import tf_util
import dataloader
from dict_restore import DictRestore
import spatial_transforms
import target_transforms
from mean import get_mean, get_std

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--data', default='', help='Data dir [default: ]')
parser.add_argument('--num_frames', type=int, default=8, help='The number of frames unsed [default: 8]')
parser.add_argument('--frame_step', type=int, default=4, help='Frame step [default: 4]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--height', type=int, default=112, help='Video image height [default: 112]')
parser.add_argument('--width', type=int, default=112, help='Video image width [default: 112]')
parser.add_argument('--num_classes', type=int, default=400, help='Number of classes [default: 400]')
parser.add_argument('--num_threads', type=int, default=24, help='Number of threads to use in loading data [default: 24]')
parser.add_argument('--fcn', type=int, default=0, help='Whether to use all spatial in evaluation [default: 0]')
parser.add_argument('--full_size', type=int, default=128, help='Full size in the shorter edge [default: 128]')
parser.add_argument('--command_file', default=None, help=' [Shell command file to use default: None]')
FLAGS = parser.parse_args()

sys.path.append(os.path.dirname(FLAGS.model_path))

random.seed(0)
np.random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

NUM_GPUS = len(FLAGS.gpu.split(','))

MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_FRAMES = FLAGS.num_frames
FRAME_STEP = FLAGS.frame_step
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
DATA = FLAGS.data
HEIGHT = FLAGS.height
WIDTH = FLAGS.width
NUM_THREADS = FLAGS.num_threads
COMMAND_FILE = FLAGS.command_file
FCN = FLAGS.fcn
FULL_SIZE = FLAGS.full_size

MODEL_FILE = os.path.join(os.path.dirname(FLAGS.model_path), FLAGS.model+'.py')
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
os.system('cp %s %s ' % (__file__, DUMP_DIR)) # bkp of evaluation file
os.system('cp %s %s ' % (COMMAND_FILE, DUMP_DIR)) # bkp of command shell file
os.system('cp %s %s' % (MODEL_FILE, DUMP_DIR)) # bkp of model def
os.system('cp utils/net_utils.py %s ' % (DUMP_DIR)) # bkp of net_utils file
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = FLAGS.num_classes

HOSTNAME = socket.gethostname()

# validation transform
normalize = spatial_transforms.ToNormalizedTensor(mean=get_mean(), std=get_std())
if FCN == 0:
    val_transform = spatial_transforms.Compose([
        spatial_transforms.Resize(FULL_SIZE),
        spatial_transforms.CenterCrop(WIDTH),
        normalize])
elif FCN == 1:
    val_transform = spatial_transforms.Compose([
        spatial_transforms.Resize(FULL_SIZE),
        spatial_transforms.CenterCrop(WIDTH),
        normalize])
elif FCN == 3:
    val_transform = spatial_transforms.Compose([
        spatial_transforms.Resize(FULL_SIZE),
        normalize])
elif FCN == 5:
    val_transform = spatial_transforms.Compose([
        spatial_transforms.Resize(FULL_SIZE),
        normalize])
elif FCN == 6:
    val_transform = spatial_transforms.Compose([
        spatial_transforms.Resize(FULL_SIZE),
        normalize])
elif FCN == 8:
    val_transform = spatial_transforms.Compose([
        spatial_transforms.Resize(FULL_SIZE),
        normalize])
elif FCN == 10:
    val_transform = spatial_transforms.Compose([
        spatial_transforms.Resize(FULL_SIZE),
        normalize])
target_transform = target_transforms.ClassLabel()

if FCN == 0:
    loader_bsize = 1
elif FCN == 1:
    loader_bsize = 10
elif FCN == 3:
    loader_bsize = 10
    WIDTH = FULL_SIZE
    HEIGHT = FULL_SIZE
elif FCN == 5:
    loader_bsize = 10
elif FCN == 6:
    loader_bsize = 8
    WIDTH = FULL_SIZE
    HEIGHT = FULL_SIZE
elif FCN == 8:
    loader_bsize = 12
    WIDTH = FULL_SIZE
    HEIGHT = FULL_SIZE
elif FCN == 10:
    loader_bsize = 25

_, val_loader = dataloader.get_loader(root=DATA, train_transform=None, val_transform=val_transform, target_transform=target_transform,
        batch_size=loader_bsize, num_frames=NUM_FRAMES, step_size=FRAME_STEP, val_samples=loader_bsize, n_threads=NUM_THREADS, training=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        is_training = False

        if FCN == 3:
            pl_bsize = 10
        elif FCN == 6:
            pl_bsize = 8
        elif FCN == 8:
            pl_bsize = 12
        elif FCN == 1:
            pl_bsize = 10
        elif FCN == 5:
            pl_bsize = 5
        elif FCN == 10:
            pl_bsize = 10
        else:
            pl_bsize = 1
        assert(pl_bsize % NUM_GPUS == 0)
        DEVICE_BATCH_SIZE = pl_bsize // NUM_GPUS

        video_pl, labels_pl = MODEL.placeholder_inputs(pl_bsize, NUM_FRAMES, HEIGHT, WIDTH, evaluate=True)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        MODEL.get_model(video_pl, is_training_pl, NUM_CLASSES)
        pred_gpu = []
        for i in range(NUM_GPUS):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d'%(i)) as scope:
                    vd_batch = tf.slice(video_pl,
                        [i*DEVICE_BATCH_SIZE,0,0,0,0], [DEVICE_BATCH_SIZE,-1,-1,-1,-1])
                    label_batch = tf.slice(labels_pl,
                        [i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])

                    pred, end_points = MODEL.get_model(vd_batch, is_training_pl, NUM_CLASSES)
                    pred_gpu.append(pred)
        pred = tf.concat(pred_gpu, 0)

        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore variables from disk.
        if MODEL_PATH is not None:
            if 'npz' not in MODEL_PATH:
                saver.restore(sess, MODEL_PATH)
                log_string("Model restored.")
            else:
                dict_file = np.load(MODEL_PATH)
                dict_for_restore = {}
                dict_file_keys = dict_file.keys()
                for k in dict_file_keys:
                    dict_for_restore[k] = dict_file[k]
                dict_for_restore = MODEL.name_mapping(dict_for_restore)
                dict_for_restore = MODEL.convert_2d_3d(dict_for_restore)
                dr = DictRestore(dict_for_restore, log_string)
                dr.run_init(sess)
                log_string("npz file restored.")

        ops = {'video_pl': video_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}

        eval_one_epoch(sess, ops, val_loader)

def eval_one_epoch(sess, ops, val_loader, topk=1):
    is_training = False

    total_correct_top1 = 0
    total_correct_top5 = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    loss_sum = 0
    batch_idx = 0
    shape_ious = []

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_data = inputs.data.numpy()
        bsize = batch_data.shape[0]
        batch_label = targets.data.numpy()
        batch_data = np.transpose(batch_data, [0,2,3,4,1])

        height = batch_data.shape[2]
        width = batch_data.shape[3]
        if FCN == 10:
            preds = []
            for i in range(25):
                batch_data_split = np.expand_dims(batch_data[i], 0)
                batch_label_split = np.expand_dims(batch_label[i], 0)
                batch_data_split = np.concatenate([ \
                        batch_data_split[:,:,:HEIGHT,:WIDTH,:], \
                        batch_data_split[:,:,:HEIGHT,:WIDTH,:][:,:,:,::-1,:], \
                        batch_data_split[:,:,-HEIGHT:,:WIDTH,:], \
                        batch_data_split[:,:,-HEIGHT:,:WIDTH,:][:,:,:,::-1,:], \
                        batch_data_split[:,:,:HEIGHT,-WIDTH:,:], \
                        batch_data_split[:,:,:HEIGHT,-WIDTH:,:][:,:,:,::-1,:], \
                        batch_data_split[:,:,-HEIGHT:,-WIDTH:,:], \
                        batch_data_split[:,:,-HEIGHT:,-WIDTH:,:][:,:,:,::-1,:], \
                        batch_data_split[:,:,(height // 2 - HEIGHT // 2):(height // 2 + HEIGHT // 2),(width // 2 - WIDTH // 2):(width // 2 + WIDTH // 2),:], \
                        batch_data_split[:,:,(height // 2 - HEIGHT // 2):(height // 2 + HEIGHT // 2),(width // 2 - WIDTH // 2):(width // 2 + WIDTH // 2),:][:,:,:,::-1,:] ], \
                        axis=0)
                batch_label_split = np.concatenate([batch_label_split] * 10, axis=0)
                feed_dict = {ops['video_pl']: batch_data_split,
                             ops['labels_pl']: batch_label_split,
                             ops['is_training_pl']: is_training}

                pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
                preds.append(pred_val)
            pred_val = np.concatenate(preds, 0)
        elif FCN == 5:
            preds = []
            for i in range(10):
                batch_data_split = np.expand_dims(batch_data[i], 0)
                batch_label_split = np.expand_dims(batch_label[i], 0)
                batch_data_split = np.concatenate([ \
                        batch_data_split[:,:,:HEIGHT,:WIDTH,:], \
                        batch_data_split[:,:,-HEIGHT:,:WIDTH,:], \
                        batch_data_split[:,:,:HEIGHT,-WIDTH:,:], \
                        batch_data_split[:,:,-HEIGHT:,-WIDTH:,:], \
                        batch_data_split[:,:,(height // 2 - HEIGHT // 2):(height // 2 + HEIGHT // 2),(width // 2 - WIDTH // 2):(width // 2 + WIDTH // 2),:]], \
                        axis=0)
                batch_label_split = np.concatenate([batch_label_split] * 5, axis=0)
                feed_dict = {ops['video_pl']: batch_data_split,
                             ops['labels_pl']: batch_label_split,
                             ops['is_training_pl']: is_training}

                pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
                preds.append(pred_val)
            pred_val = np.concatenate(preds, 0)
        else:
            if FCN == 3:
                if height > width:
                    assert(width == FULL_SIZE)
                    batch_data_list = [ batch_data[:,:,:FULL_SIZE,:,:], \
                                        batch_data[:,:,-FULL_SIZE:,:,:], \
                                        batch_data[:,:,(height // 2 - FULL_SIZE // 2):(height // 2 + FULL_SIZE // 2),:,:]]
                else:
                    assert(height == FULL_SIZE)
                    batch_data_list = [ batch_data[:,:,:,:FULL_SIZE,:], \
                                        batch_data[:,:,:,-FULL_SIZE:,:], \
                                        batch_data[:,:,:,(width // 2 - FULL_SIZE // 2):(width // 2 + FULL_SIZE // 2),:]]
                batch_label_list = [batch_label] * 3
                preds = []
                for i in range(3):
                    feed_dict = {ops['video_pl']: batch_data_list[i],
                                 ops['labels_pl']: batch_label_list[i],
                                 ops['is_training_pl']: is_training}
                    pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
                    preds.append(pred_val)
                pred_val = np.concatenate(preds, 0)
            elif FCN == 6:
                if height > width:
                    assert(width == FULL_SIZE)
                    batch_data_list = [ batch_data[:,:,:FULL_SIZE,:,:], \
                                        batch_data[:,:,-FULL_SIZE:,:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 5 * 1):((height-FULL_SIZE) // 5 * 1 + FULL_SIZE),:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 5 * 2):((height-FULL_SIZE) // 5 * 2 + FULL_SIZE),:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 5 * 3):((height-FULL_SIZE) // 5 * 3 + FULL_SIZE),:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 5 * 4):((height-FULL_SIZE) // 5 * 4 + FULL_SIZE),:,:], \
                                        ]
                else:
                    assert(height == FULL_SIZE)
                    batch_data_list = [ batch_data[:,:,:,:FULL_SIZE,:], \
                                        batch_data[:,:,:,-FULL_SIZE:,:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 5 * 1):((width-FULL_SIZE) // 5 * 1 + FULL_SIZE),:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 5 * 2):((width-FULL_SIZE) // 5 * 2 + FULL_SIZE),:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 5 * 3):((width-FULL_SIZE) // 5 * 3 + FULL_SIZE),:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 5 * 4):((width-FULL_SIZE) // 5 * 4 + FULL_SIZE),:], \
                                        ]
                batch_label_list = [batch_label] * 6
                preds = []
                for i in range(6):
                    feed_dict = {ops['video_pl']: batch_data_list[i],
                                 ops['labels_pl']: batch_label_list[i],
                                 ops['is_training_pl']: is_training}
                    pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
                    preds.append(pred_val)
                pred_val = np.concatenate(preds, 0)
            elif FCN == 8:
                if height > width:
                    assert(width == FULL_SIZE)
                    batch_data_list = [ batch_data[:,:,:FULL_SIZE,:,:], \
                                        batch_data[:,:,-FULL_SIZE:,:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 7 * 1):((height-FULL_SIZE) // 7 * 1 + FULL_SIZE),:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 7 * 2):((height-FULL_SIZE) // 7 * 2 + FULL_SIZE),:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 7 * 3):((height-FULL_SIZE) // 7 * 3 + FULL_SIZE),:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 7 * 4):((height-FULL_SIZE) // 7 * 4 + FULL_SIZE),:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 7 * 5):((height-FULL_SIZE) // 7 * 5 + FULL_SIZE),:,:], \
                                        batch_data[:,:,((height-FULL_SIZE) // 7 * 6):((height-FULL_SIZE) // 7 * 6 + FULL_SIZE),:,:], \
                                        ]
                else:
                    assert(height == FULL_SIZE)
                    batch_data_list = [ batch_data[:,:,:,:FULL_SIZE,:], \
                                        batch_data[:,:,:,-FULL_SIZE:,:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 7 * 1):((width-FULL_SIZE) // 7 * 1 + FULL_SIZE),:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 7 * 2):((width-FULL_SIZE) // 7 * 2 + FULL_SIZE),:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 7 * 3):((width-FULL_SIZE) // 7 * 3 + FULL_SIZE),:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 7 * 4):((width-FULL_SIZE) // 7 * 4 + FULL_SIZE),:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 7 * 5):((width-FULL_SIZE) // 7 * 5 + FULL_SIZE),:], \
                                        batch_data[:,:,:,((width-FULL_SIZE) // 7 * 6):((width-FULL_SIZE) // 7 * 6 + FULL_SIZE),:], \
                                        ]
                batch_label_list = [batch_label] * 8
                preds = []
                for i in range(8):
                    feed_dict = {ops['video_pl']: batch_data_list[i],
                                 ops['labels_pl']: batch_label_list[i],
                                 ops['is_training_pl']: is_training}
                    pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
                    preds.append(pred_val)
                pred_val = np.concatenate(preds, 0)
            else:
                feed_dict = {ops['video_pl']: batch_data,
                             ops['labels_pl']: batch_label,
                             ops['is_training_pl']: is_training}

                pred_val = sess.run(ops['pred'], feed_dict=feed_dict)

        pred_val = np.exp(pred_val - np.max(pred_val, axis=1, keepdims=True))
        pred_val = pred_val / np.sum(pred_val, axis=1, keepdims=True)

        pred_val = np.mean(pred_val, axis=0)

        pred_val_top5 = np.argsort(pred_val)[::-1][:5]
        pred_val_top1 = np.argmax(pred_val)

        correct_top1 = pred_val_top1 == batch_label[0]
        correct_top5 = np.sum(pred_val_top5 == batch_label[0])
        total_correct_top1 += correct_top1
        total_correct_top5 += correct_top5

        total_seen += 1
        log_string('batch accuracy top1 : %f, batch accuracy top5: %f'% (correct_top1, correct_top5))

        l = batch_label[0]
        total_seen_class[l] += 1
        total_correct_class[l] += correct_top1

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy top1 : %f' % (total_correct_top1 / float(total_seen)))
    log_string('eval accuracy top5 : %f' % (total_correct_top5 / float(total_seen)))
    np.savez_compressed(os.path.join(DUMP_DIR, 'pred_class.npz'), total_correct_class=np.array(total_correct_class), total_seen_class=np.array(total_seen_class))


if __name__=='__main__':
    evaluate()
    LOG_FOUT.close()
