#!/bin/bash
set -x #echo on

export PYTHONPATH=$PYTHONPATH:$1

python3 -m gulpio.GulpAdapter \
    --json_file /datasets/jester/val_labels.json \
    --input_path /datasets/jester/mp4_cat/val \
    --output_path /datasets/jester/gulp_128/val \
    --num_workers 4 \
    --frame_size 128 \
    --frame_rate 12 \
    --skip_exist

python3 -m gulpio.GulpAdapter \
    --json_file /datasets/jester/train_labels.json \
    --input_path /datasets/jester/mp4_cat/train \
    --output_path /datasets/jester/gulp_128/train \
    --num_workers 4 \
    --frame_size 128 \
    --frame_rate 12 \
    --skip_exist

python3 -m gulpio.GulpAdapter \
    --json_file /datasets/jester/test_labels.json \
    --input_path /datasets/jester/mp4_cat/test \
    --output_path /datasets/jester/gulp_128/test \
    --num_workers 4 \
    --frame_size 128 \
    --frame_rate 12 \
    --skip_exist
