


import os
import csv
import json
import re


label_json = '/mnt/ssd/tmp/something/gulp_128/train/label2idx.json'
test_file = 'log_train_resnet18_pointnet_224_step_2_test_3/log_test.txt'
csv_file = 'log_train_resnet18_pointnet_224_step_2_test_3/test_result.csv'

with open(label_json, 'r') as jf:
    label_to_id = json.loads(jf.read())

id_to_label = {}
labels = label_to_id.keys()
for l in labels:
    id_to_label[str(label_to_id[l])] = l

csv_list = []
with open(test_file, 'r') as t:
    l = t.readline()
    while len(l) > 0:
        l = l.rstrip()

        if '[' in l:
            vid = l.split(':')[0]
            top5_string = l.split('[')[1].split(']')[0]
            top5 = re.split(' ', top5_string)
            top5 = [t for t in top5 if len(t) > 0]
            assert(len(top5) == 5)
            csv_list.append([vid, top5[0], top5[1], top5[2], top5[3], top5[4]])

        l = t.readline()

with open(csv_file, 'w') as cf:
    writer = csv.writer(cf, delimiter=';')
    for row in csv_list:
        writer.writerow(row)

