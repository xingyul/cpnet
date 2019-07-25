


import os
import glob
import json
import json
import argparse
import multiprocessing

from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser()
parser.add_argument('--mp4_dir', default='/datasets/something-something/mp4', type=str, help='mp4 root dir')
parser.add_argument('--mp4_cat_dir', default='/datasets/something-something/mp4_cat', type=str, help='mp4 categorized root dir')
parser.add_argument('--input_json_dir', default='.', type=str, help='input json dir')
parser.add_argument('--output_json_dir', default='/datasets/something-something', type=str, help='output json dir')
FLAGS = parser.parse_args()

mp4_dir = FLAGS.mp4_dir
mp4_cat_dir = FLAGS.mp4_cat_dir
input_json_dir = FLAGS.input_json_dir
output_json_dir = FLAGS.output_json_dir

train_json_file = os.path.join(input_json_dir, 'something-something-v2-train.json')
val_json_file = os.path.join(input_json_dir, 'something-something-v2-validation.json')
test_json_file = os.path.join(input_json_dir, 'something-something-v2-test.json')

def transform_str(string_):
    string = string_.replace('[', '')
    string = string.replace(']', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('\'', '')
    return string

for json_filename, div in [[train_json_file, 'train'], [val_json_file, 'val']]:
    with open(json_filename, 'r') as jsonfile:
        data = json.load(jsonfile)
        all_jsons = {}
        for d in data:
            print(d['id'])
            try:
                clip = VideoFileClip(os.path.join(mp4_dir, d['id'] + '.mp4'))
                duration = clip.duration
                all_jsons[d['id']] = {'annotations': {'label': transform_str(d['template']), 'segment': [0, duration]}, 'duration': duration, 'subset': div}
            except:
                print('bad:', d['id'])

        with open(os.path.join(output_json_dir, div + '_labels.json'), 'w') as outfile:
            outfile.write(json.dumps(all_jsons, sort_keys=True, indent=4))

with open(test_json_file, 'r') as jsonfile:
    data = json.load(jsonfile)
    all_jsons = {}
    for d in data:
        print(d['id'])
        try:
            clip = VideoFileClip(os.path.join(mp4_dir, d['id'] + '.mp4'))
            duration = clip.duration
            all_jsons[d['id']] = {'annotations': {'label': '0', 'segment': [0, duration]}, 'duration': duration, 'subset': 'test'}
        except:
            print('bad:', d['id'])

    with open(os.path.join(output_json_dir, 'test_labels.json'), 'w') as outfile:
        outfile.write(json.dumps(all_jsons, sort_keys=True, indent=4))




