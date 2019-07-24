



### Data Preprocessing Guideline

#### Data Download

To process the raw data, first download <a href="https://20bn.com/datasets/something-something">Something-Something dataset</a>. Then extract the files in, for example `/raid/datasets/something-something/20bn-something-something-v2`, such that the directory looks like

```
/raid/datasets/something-something/
  20bn-something-something-v2/
    1.webm
    2.webm
    220847.webm
```

#### Processing Steps

`cd` into directory `utils/data_preparation/something-something`. Suppose the default  working directory containing the output processed files is `/datasets/something-something/gulp_240`, then execute commands to generate gulped files of video data. 

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

The output processed data directory should look like

```
/datasets/something-something/gulp_240/
  train/
    Approaching something with your camera/
      xxxxx.gmeta
      xxxxx.gulp
      ...
    Attaching something to something/
      xxxxx.gmeta
      xxxxx.gulp
      ...
    ...
    label2idx.json
    gulp_log.csv
    opts.json
  val/
    Approaching something with your camera/
      xxxxx.gmeta
      xxxxx.gulp
      ...
    Attaching something to something/
      xxxxx.gmeta
      xxxxx.gulp
      ...
    ...
    label2idx.json
    gulp_log.csv
    opts.json
  test/
    0/
      xxxxx.gmeta
      xxxxx.gulp
      ...
    label2idx.json
    gulp_log.csv
    opts.json
```

