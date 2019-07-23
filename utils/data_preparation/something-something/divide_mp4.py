


import os
import glob
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mp4_src_dir', default='/datasets/something-something/mp4', type=str, help='mp4 input root dir')
parser.add_argument('--mp4_cat_dir', default='/datasets/something-something/mp4_cat', type=str, help='categorized mp4 output dir')
parser.add_argument('--json_dir', default='.', type=str, help='json dir')
FLAGS = parser.parse_args()

mp4_src_dir = FLAGS.mp4_src_dir
mp4_cat_dir = FLAGS.mp4_cat_dir
json_dir = FLAGS.json_dir

if not os.path.exists(mp4_cat_dir):
    os.system('mkdir -p {}'.format(mp4_cat_dir))

train_json_file = os.path.join(json_dir, 'something-something-v2-train.json')
val_json_file = os.path.join(json_dir, 'something-something-v2-validation.json')
test_json_file = os.path.join(json_dir, 'something-something-v2-test.json')

def transform_cmd_line_str(string_):
    string = string_.replace('[', '')
    string = string.replace(']', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('\'', '')
    string = string.replace(' ', '\ ')
    return string

'''
for json_filename, div in [[train_json_file, 'train'], [val_json_file, 'val']]:
    with open(json_filename, 'r') as jsonfile:
        data = json.load(jsonfile)
        for d in data:
            out_dir_ = os.path.join(mp4_cat_dir, div, d['template'])
            out_dir_ = transform_cmd_line_str(out_dir_)
            if not os.path.exists(out_dir_):
                os.system('mkdir -p {}'.format(out_dir_))
            os.system('ln -s {} {}'.format(os.path.join(mp4_src_dir, d['id'] + '.mp4'), out_dir_))
'''

with open(test_json_file, 'r') as jsonfile:
    data = json.load(jsonfile)
    for d in data:
        out_dir_ = os.path.join(mp4_cat_dir, 'test', '0')
        out_dir_ = transform_cmd_line_str(out_dir_)
        if not os.path.exists(out_dir_):
            os.system('mkdir -p {}'.format(out_dir_))
        os.system('ln -s {} {}'.format(os.path.join(mp4_src_dir, d['id'] + '.mp4'), out_dir_))

