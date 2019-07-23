





Convert `.jpg` files to `.mp4` files.
```
python jpg_to_mp4.py \
    --jpg_dir /raid/datasets/jester/20bn-jester-v1 \
    --mp4_dir /datasets/jester/mp4
```

Download csv files from https://20bn.com/datasets/jester/.
Put `jester-v1-test.csv`, `jester-v1-train.csv` and 
`jester-v1-validation.csv` in this directory.

Split `.mp4` files into `train`, `val` and `test` splits.
```
python divide_mp4.py \
    --mp4_src_dir /datasets/jester/mp4 \
    --mp4_cat_dir /datasets/jester/mp4_cat \
    --csv_dir .
```

Then generate `.json` files.
```
python gen_json.py --mp4_dir /datasets/jester/mp4_cat \
    --csv_dir .
    --json_dir /datasets/jester 
```

Generate gulp files
```
sh gen_gulp.sh /abs/path/to/cpnet/utils
```


