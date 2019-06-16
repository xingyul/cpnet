#!/usr/bin/env python

import os
import sh
import random
import cv2
import shutil
import glob
from contextlib import contextmanager

###############################################################################
#                                Helper Functions                             #
###############################################################################


class FFMPEGNotFound(Exception):
    pass


def check_ffmpeg_exists():
    return os.system('ffmpeg -version > /dev/null') == 0


@contextmanager
def temp_dir_for_bursting(shm_dir_path='/dev/shm'):
    hash_str = str(random.getrandbits(128))
    temp_dir = os.path.join(shm_dir_path, hash_str)
    os.makedirs(temp_dir)  # creates error if paths conflict (unlikely)
    yield temp_dir
    shutil.rmtree(temp_dir)


def burst_frames_to_shm(vid_path, temp_burst_dir, frame_size=-1, frame_rate=-1, frame_uniform=-1):
    """
    - To burst frames in a temporary directory in shared memory.
    - Directory name is chosen as random 128 bits so as to avoid clash
      during parallelization
    - Returns path to directory containing frames for the specific video
    """
    target_mask = os.path.join(temp_burst_dir, '%06d.bmp')
    if not os.system('ffmpeg -version > /dev/null') == 0:
        raise FFMPEGNotFound()
    try:
        if frame_size > 0 or frame_uniform > 0:
            ffprobe_args = [
                '-v', 'error',
                '-count_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_read_frames',
                vid_path
            ]
            output = sh.ffprobe(ffprobe_args)
            [width_src, height_src, total_frames] = [int(o) for o in output.split()[-3:]]

            vf_param = []
            if frame_size > 0:
                if height_src < width_src:
                    vf_param.append('scale=-1:{}'.format(int(frame_size)))
                else:
                    vf_param.append('scale={}:-1'.format(int(frame_size)))
            if frame_uniform > 0:
                if frame_uniform == 1:  # select the middle frame
                    frames_to_sample = [(total_frames+1) // 2]
                else:
                    frames_to_sample = np.linspace(1,total_frames-1,frame_uniform).round().astype(int)
                select_param = '+'.join( [ 'eq(n\,{})'.format(n) for n in frames_to_sample] )
                vf_param.append('select=' + select_param)

        ffmpeg_args = [
            '-i', vid_path,
            '-q:v', str(1),
            '-f', 'image2',
            target_mask,
        ]
        if frame_size > 0 or frame_uniform > 0:
            ffmpeg_args.insert(2, '-vf')
            ffmpeg_args.insert(3, ','.join(vf_param))
            if frame_uniform > 0:
                ffmpeg_args.insert(4, '-vsync')
                ffmpeg_args.insert(5, 'vfr')
        if frame_rate > 0:
            ffmpeg_args.insert(2, '-r')
            ffmpeg_args.insert(3, str(frame_rate))

        sh.ffmpeg(*ffmpeg_args)
    except Exception as e:
        pass
        #print(repr(e))

def burst_video_into_frames(vid_path, temp_burst_dir, frame_size=-1, frame_rate=-1, frame_uniform=-1):
    burst_frames_to_shm(vid_path, temp_burst_dir, frame_size=frame_size,
                        frame_rate=frame_rate, frame_uniform=frame_uniform)
    return find_images_in_folder(temp_burst_dir, formats=['bmp'])



class ImageNotFound(Exception):
    pass


class DuplicateIdException(Exception):
    pass


def cvt_bgr2rgb(imgs):
    for img in imgs:
        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_images(imgs, img_size=-1):
    for img in imgs:
        img_path = img
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        if img is None:
            raise ImageNotFound("Image is  None from path:{}".format(img_path))
        if img_size > 0:
            img = resize_by_short_edge(img, img_size)
        yield img


def resize_by_short_edge(img, size):
    if isinstance(img, str):
        img_path = img
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        if img is None:
            raise ImageNotFound("Image read None from path ", img_path)
    if size < 1:
        return img
    h, w = img.shape[0], img.shape[1]
    if h < w:
        scale = w / float(h)
        new_width = int(size * scale)
        img = cv2.resize(img, (new_width, size))
    else:
        scale = h / float(w)
        new_height = int(size * scale)
        img = cv2.resize(img, (size, new_height))
    return img


###############################################################################
#                       Helper Functions for input iterator                   #
###############################################################################

def find_images_in_folder(folder, formats=['jpg']):
    images = []
    for format_ in formats:
        files = glob.glob('{}/*.{}'.format(folder, format_))
        images.extend(files)
    return sorted(images)

def find_files_in_subfolders(folder, formats=['mp4']):
    formats_new = []
    for format_ in formats:
        if format_[0] == '.':
            formats_new.append(format_)
        else:
            formats_new.append('.' + format_)
    videos = []
    for root, dirs, files in os.walk(folder, followlinks=True):
        for f in files:
            fullpath = os.path.join(root, f)
            for format_ in formats_new:
                if fullpath.endswith(format_):
                    videos.append(fullpath)
    return sorted(videos)


###############################################################################
#                       Helper Functions for sanity check                   #
###############################################################################
def is_chunk_valid(chunk):
    try:
        if os.path.exists(chunk.data_file_path) and os.path.exists(chunk.meta_file_path):
            data_file_size = os.stat(chunk.data_file_path).st_size
            meta_file_size = os.stat(chunk.meta_file_path).st_size
            if data_file_size > 0 and meta_file_size > 0:
                if len(chunk.meta_dict[list(chunk.meta_dict.keys())[-1]]['meta_data']) == 1:
                    frame_info = chunk.meta_dict[next(reversed(chunk.meta_dict))]['frame_info']
                    num_frames = len(frame_info)
                    last_frame_info = frame_info[-1]
                    data_file_size_from_meta = last_frame_info[0] + last_frame_info[2]
                    if data_file_size == data_file_size_from_meta:
                        return num_frames
    except:
        pass
    return -1
