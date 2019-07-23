#!/bin/bash
set -x #echo on

export PYTHONPATH=$PYTHONPATH:$1

python3 -m gulpio.GulpAdapter \
    --json_file /datasets/something-something/val_labels.json \
    --input_path /datasets/something-something/mp4_cat/val \
    --output_path /datasets/something-something/gulp_240/val \
    --num_workers 4 \
    --frame_size 240 \
    --frame_rate 24 \
    --skip_exist

python3 -m gulpio.GulpAdapter \
    --json_file /datasets/something-something/train_labels.json \
    --input_path /datasets/something-something/mp4_cat/train \
    --output_path /datasets/something-something/gulp_240/train \
    --num_workers 4 \
    --frame_size 240 \
    --frame_rate 24 \
    --skip_exist

python3 -m gulpio.GulpAdapter \
    --json_file /datasets/something-something/test_labels.json \
    --input_path /datasets/something-something/mp4_cat/test \
    --output_path /datasets/something-something/gulp_240/test \
    --num_workers 4 \
    --frame_size 240 \
    --frame_rate 24 \
    --skip_exist
