


import os
import glob
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mp4_src_dir', default='/datasets/jester/mp4', type=str, help='mp4 input root dir')
parser.add_argument('--mp4_cat_dir', default='/datasets/jester/mp4_cat', type=str, help='categorized mp4 output dir')
parser.add_argument('--csv_dir', default='.', type=str, help='csv dir')
FLAGS = parser.parse_args()

mp4_src_dir = FLAGS.mp4_src_dir
mp4_cat_dir = FLAGS.mp4_cat_dir
csv_dir = FLAGS.csv_dir

if not os.path.exists(mp4_cat_dir):
    os.system('mkdir -p {}'.format(mp4_cat_dir))

train_csv_file = os.path.join(csv_dir, 'jester-v1-train.csv')
val_csv_file = os.path.join(csv_dir, 'jester-v1-validation.csv')
test_csv_file = os.path.join(csv_dir, 'jester-v1-test.csv')


for csv_filename, div in [[train_csv_file, 'train'], [val_csv_file, 'val']]:
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for r in reader:
            out_dir_ = os.path.join(mp4_cat_dir, div, r[1])
            out_dir = out_dir_.replace(' ', '\ ')
            if not os.path.exists(out_dir_):
                os.system('mkdir -p {}'.format(out_dir))
            os.system('ln -s {} {}'.format(os.path.join(mp4_src_dir, r[0] + '.mp4'), out_dir))

with open(test_csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for r in reader:
        out_dir_ = os.path.join(mp4_cat_dir, 'test', '0')
        out_dir = out_dir_.replace(' ', '\ ')
        if not os.path.exists(out_dir_):
            os.system('mkdir -p {}'.format(out_dir))
        os.system('ln -s {} {}'.format(os.path.join(mp4_src_dir, r[0] + '.mp4'), out_dir))

