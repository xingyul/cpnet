

import os
import glob
import cv2
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--jpg_dir', default='/raid/datasets/jester/20bn-jester-v1', type=str, help='jpg input root dir')
parser.add_argument('--mp4_dir', default='/datasets/jester/mp4', type=str, help='mp4 output dir')
FLAGS = parser.parse_args()

jpg_dir = FLAGS.jpg_dir
mp4_dir = FLAGS.mp4_dir

if not os.path.exists(mp4_dir):
    os.system('mkdir -p {}'.format(mp4_dir))

videos = glob.glob(os.path.join(jpg_dir, '*'))

def process_video(v):
    vid = os.path.basename(v)
    jpg = os.path.join(v, '00001.jpg')
    im = cv2.imread(jpg)
    h, w = im.shape[:2]

    output_file = os.path.join(mp4_dir, vid + '.mp4')

    jpgs = os.path.join(v, '%05d.jpg')
    os.system('ffmpeg -r 12 -f image2 -s {}x{} -i {} -crf 25 {}'.format(w, h, jpgs, output_file))

pool = multiprocessing.Pool(processes=8)

for v in videos:
    print(v)
    pool.apply_async(process_video, (v,))

pool.close()
pool.join()
