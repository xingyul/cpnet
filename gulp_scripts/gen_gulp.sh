#!/bin/bash
set -x #echo on

python -m gulpio.GulpAdapter \
    --json_file /mnt/ssd/tmp/20bn-something-something-v2/val_label.json \
    --input_path /mnt/ssd/tmp/20bn-something-something-v2/mp4_category/val \
    --output_path /mnt/ssd/tmp/20bn-something-something-v2/gulp_128/val \
    --num_workers 24 \
    --frame_size 128 \
    --frame_rate 24 \
    --skip_exist

python -m gulpio.GulpAdapter \
    --json_file /mnt/ssd/tmp/20bn-something-something-v2/train_label.json \
    --input_path /mnt/ssd/tmp/20bn-something-something-v2/mp4_category/train \
    --output_path /mnt/ssd/tmp/20bn-something-something-v2/gulp_128/train \
    --num_workers 24 \
    --frame_size 128 \
    --frame_rate 24 \
    --skip_exist
