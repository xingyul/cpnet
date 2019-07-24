



### Data Preprocessing Guideline

#### Data Download

To process the raw data, first download <a href="https://20bn.com/datasets/jester">Jester dataset</a>. Then extract the files in, for example `/raid/datasets/jester/20bn-jester-v1`, such that the directory looks like

```
/raid/datasets/jester/
  20bn-jester-v1
    1/
    2/
    ...
    148092/
  jester-v1-test.csv
  jester-v1-train.csv
  jester-v1-validation.csv
```

#### Processing Steps

`cd` into directory `utils/data_preparation/jester`. Suppose the default directory containing the output processed files is `/datasets/jester/gulp_128`, then execute the following commands to generate gulped files of video data. 

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

The output processed data directory should look like

```
/datasets/jester/gulp_128/
  train/
    Doing other things/
      100018.gmeta
      100018.gulp
      ...
    Drumming Fingers/
      100022.gmeta
      100022.gulp
      ...
    ...
    label2idx.json
    gulp_log.csv
    opts.json
  val/
    Doing other things/
      100090.gmeta
      100090.gulp
      ...
    Drumming Fingers/
      100001.gmeta
      100001.gulp
      ...
    ...
    label2idx.json
    gulp_log.csv
    opts.json
  test/
    0/
      100005.gmeta
      100005.gulp
      ...
    label2idx.json
    gulp_log.csv
    opts.json
```

