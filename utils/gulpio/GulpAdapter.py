import os
import sys
import json
import glob
import sh
import random
import argparse
import gulpio.gulpio_utils as gutils
from gulpio.gulpio import AbstractDatasetAdapter, GulpIngestor


class GulpAdapter(AbstractDatasetAdapter):
    """
    An Adapter for the Video dataset (trimmed).
    This adapter assumes that the file name of each video has the
    following patterns:
        (vid)_(start)_(end).mp4
    where
      vid: video id that was assigned by youtube
      start: the beginning of trimming in the original youtube video
             in 6 digit-second
      end: the end of trimming in the original youtube video
             in 6 digit-second
    """
    def __init__(self, json_file, input_path, output_path,
                 frame_size=-1, frame_rate=-1, shm_dir_path='/dev/shm'):
        self.input_path = input_path
        self.output_path = output_path
        with open(json_file, 'r') as f:
            self.json_storage = json.load(f)
        self.write_label2idx_dict()
        self.data = self.get_video_list()
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.shm_dir_path = shm_dir_path

    def __len__(self):
        return len(self.data)

    def get_chunk_path(self, index):
        video_file = self.data[index]
        vid_id = video_file.split('/')[-1].split('.')[0]
        meta_json = self.json_storage.get(vid_id)
        chunk_path = os.path.join(meta_json['annotations']['label'], vid_id)
        return chunk_path

    def get_data(self, index):
        video_file = self.data[index]
        frames = self.get_bursted_frames(video_file)
        # this is due to current file names of trimmed videos
        vid_id = video_file.split('/')[-1].split('.')[0]
        meta_json = self.json_storage.get(vid_id)
        meta = {'label': meta_json['annotations']['label'],
                'subset': meta_json['subset']}
        result = {'meta': meta,
                    'frames': frames,
                    'id': vid_id}
        return result

    def write_label2idx_dict(self):
        labels = sorted(set([meta['annotations']['label'] for video_id, meta in self.json_storage.items()]))
        label2idx = {label: label_counter for label_counter, label in enumerate(labels)}
        json.dump(label2idx,
                  open(os.path.join(self.output_path, 'label2idx.json'), 'w'))

    def get_video_list(self):
        video_list = gutils.find_files_in_subfolders(self.input_path, ['mp4'])
        # remove video if there is no corresponding json item
        video_list = [v for v in video_list
                        if self.json_storage.get(v.split('/')[-1].split('.')[0])]
        return sorted(video_list)

    def get_bursted_frames(self, vid_file):
        vid_file = os.path.join(self.input_path, vid_file)
        with gutils.temp_dir_for_bursting(self.shm_dir_path) as temp_burst_dir:
            frame_paths = gutils.burst_video_into_frames(vid_file, temp_burst_dir, frame_rate=self.frame_rate)
            frames = list(gutils.cvt_bgr2rgb(gutils.resize_images(frame_paths, self.frame_size)))
        return frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, help='File path of JSON')
    parser.add_argument('--input_path', type=str, help='Directory path of input videos')
    parser.add_argument('--output_path', type=str, help='Output directory path')
    parser.add_argument('--num_workers', default=16, type=int,help='Number of workers for parallel processing')
    parser.add_argument('--frame_size', default=-1, type=int,help='Shorter edge size for image resizing')
    parser.add_argument('--frame_rate', default=-1, type=int,help='frame rate for frame extraction')
    parser.add_argument('--skip_exist', action='store_true', help='Skip if exists')
    parser.add_argument('--shm_dir_path', default='/dev/shm', type=str, help='shared memory path')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    with open(os.path.join(opt.output_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file, indent=4)

    print('Compute file list...')
    adapter = GulpAdapter(opt.json_file, opt.input_path, opt.output_path,
                            frame_size=opt.frame_size, frame_rate=opt.frame_rate, shm_dir_path=opt.shm_dir_path)

    print('Start Gulping...')
    ingestor = GulpIngestor(adapter, opt.output_path, opt.num_workers, opt.skip_exist)
    ingestor()
