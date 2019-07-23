

import os
import glob
import cv2
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--webm_dir', default='/raid/datasets/something-something/20bn-something-something-v2', type=str, help='webm input root dir')
parser.add_argument('--mp4_dir', default='/datasets/something-something/mp4', type=str, help='mp4 output dir')
FLAGS = parser.parse_args()

webm_dir = FLAGS.webm_dir
mp4_dir = FLAGS.mp4_dir

if not os.path.exists(mp4_dir):
    os.system('mkdir -p {}'.format(mp4_dir))

videos = glob.glob(os.path.join(webm_dir, '*'))

def process_video(v):
    vid = os.path.basename(v).split('.webm')[0]
    output_file = os.path.join(mp4_dir, vid + '.mp4')

    video = cv2.VideoCapture(v)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

    height = (int(height) // 2) * 2
    width = (int(width) // 2) * 2

    os.system('ffmpeg -i {} -s {}x{} {}'.format(v, width, height, output_file))

pool = multiprocessing.Pool(processes=8)

for v in videos:
    print(v)
    pool.apply_async(process_video, (v,))

pool.close()
pool.join()
