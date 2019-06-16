


import os
import glob
import csv
import json
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--mp4_dir', default='/datasets/jester/mp4_cat', type=str, help='mp4 categorized root dir')
parser.add_argument('--csv_dir', default='/raid/datasets/jester/', type=str, help='csv dir')
parser.add_argument('--json_dir', default='/datasets/jester', type=str, help='output json dir')
FLAGS = parser.parse_args()

mp4_dir = FLAGS.mp4_dir
csv_dir = FLAGS.csv_dir
json_dir = FLAGS.json_dir

train_csv_file = os.path.join(csv_dir, 'jester-v1-train.csv')
val_csv_file = os.path.join(csv_dir, 'jester-v1-validation.csv')
test_csv_file = os.path.join(csv_dir, 'jester-v1-test.csv')


for csv_filename, div in [[train_csv_file, 'train'], [val_csv_file, 'val']]:
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        all_jsons = {}
        for r in reader:
            print(r)
            try:
                duration = 3.
                all_jsons[r[0]] = {'annotations': {'label': r[1], 'segment': [0, duration]}, 'duration': duration, 'subset': div}
            except:
                print('bad:', r)

        with open(os.path.join(json_dir, div + '_labels.json'), 'w') as outfile:
            outfile.write(json.dumps(all_jsons, sort_keys=True, indent=4))

with open(test_csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    all_jsons = {}
    for r in reader:
        print(r)
        try:
            duration = 3.
            all_jsons[r[0]] = {'annotations': {'label': '0', 'segment': [0, duration]}, 'duration': duration, 'subset': 'test'}
        except:
            print('bad:', r)

    with open(os.path.join(json_dir, 'test_labels.json'), 'w') as outfile:
        outfile.write(json.dumps(all_jsons, sort_keys=True, indent=4))




