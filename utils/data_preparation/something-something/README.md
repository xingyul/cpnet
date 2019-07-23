





Convert `.webm` files to `.mp4` files.
```
python webm_to_mp4.py \
    --jpg_dir /raid/datasets/something-something/20bn-something-something-v2 \
    --mp4_dir /datasets/something-something/mp4
```

Download json files from https://20bn.com/datasets/something-something/.
Put `something-something-v2-test.json`, `something-something-v2-train.json`,
`something-something-v2-validation.json` and `something-something-v2-labels.json` in this directory.

Split `.mp4` files into `train`, `val` and `test` splits.
```
python divide_mp4.py \
    --mp4_src_dir /datasets/something-something/mp4 \
    --mp4_cat_dir /datasets/something-something/mp4_cat \
    --json_dir .
```

Then generate `.json` files.
```
python gen_json.py \
    --mp4_dir /datasets/something-something/mp4 \
    --mp4_cat_dir /datasets/something-something/mp4_cat \
    --input_json_dir . \
    --output_json_dir /datasets/something-something
```

Generate gulp files
```
sh gen_gulp.sh /abs/path/to/cpnet/utils
```


