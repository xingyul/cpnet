### Learning Video Representations from Correspondence Proposals
Created by <a href="http://xingyul.github.io">Xingyu Liu</a>, <a href="https://joonyoung-cv.github.io" target="_blank">Joon-Young Lee</a> and <a href="https://sites.google.com/view/hailinjin" target="_blank">Hailin Jin</a> from Stanford University and Adobe Research.

<img src="https://github.com/xingyul/cpnet/blob/master/doc/teaser.png" width="80%">

### Citation
If you find our work useful in your research, please cite:

        @article{liu:2019:cpnet,
          title={Learning Video Representations from Correspondence Proposals},
          author={Xingyu Liu and Joon-Young Lee and Hailin Jin},
          journal={CVPR},
          year={2019}
        }

### Abstract

Correspondences between frames encode rich information about dynamic content in videos. However, it is challenging to effectively capture and learn those due to their irregular structure and complex dynamics. In this paper, we propose a novel neural network that learns video representations by aggregating information from potential correspondences. This network, named CPNet, can learn evolving 2D fields with temporal consistency. In particular, it can effectively learn representations for videos by mixing appearance and long-range motion with an RGB-only input. We provide extensive ablation experiments to validate our model. CPNet shows stronger performance than existing methods on Kinetics and achieves the state-of-the-art performance on Something-Something and Jester. We provide analysis towards the behavior of our model and show its robustness to errors in proposals.

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.9.0 GPU version, g++ 5.4.0, CUDA 9.0 and Python 3.5 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`. It's highly recommended that you have access to GPUs.

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them first by `make` under each ops subfolder (check `Makefile`). Update `arch` in the Makefiles for different <a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported">CUDA Compute Capability</a> that suits your GPU if necessary.

### Usage

#### Jester Experiments

The data preprocessing scripts are included in `utils/data_preparation`. To process the raw data, first download <a href="https://20bn.com/datasets/jester">Jester dataset</a>. Then extract the files in, for example `/raid/datasets/jester/20bn-jester-v1`, such that the directory looks like

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

Then `cd` into directory `utils/data_preparation/jester`. Suppose the default directory containing the output processed files is `/datasets/jester/gulp_128`, then execute commands following the `README.md` in that directory to generate gulped files of video data. For other output directories other than the default one, directories in `utils/data_preparation/jester/gen_gulp.sh` will also need to be changed. The output processed data directory should look like

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

#### Training and Evaluation

First download the ImageNet pretrained ResNet model from <a href="http://models.tensorpack.com/ResNet/">here</a> and put it in `pretrained_models/ImageNet-ResNet34.npz`.

To train the model, rename `command_train.sh.jester.experiment` to be `command_train.sh` and simply execute the shell script `command_train.sh`. Batch size, learning rate etc are adjustable.

```
sh command_train.sh
```

To evaluate the model, rename `command_evaluate.sh.jester.experiment` to be `command_evaluate.sh` and simply execute the shell script `command_evaluate.sh`.

```
sh command_evaluate.sh
```

To test the model, rename `command_test.sh.jester.experiment` to be `command_test.sh` and simply execute the shell script `command_test.sh`.

```
sh command_test.sh
```

#### Something-Something Experiment

Similar to Jester experiment. To be released. Stay Tuned.

### License
Our code is released under CC BY-NC-SA-4.0 License (see LICENSE file for details).

### Related Projects

* <a href="https://arxiv.org/abs/1806.01411" target="_blank">FlowNet3D: Learning Scene Flow in 3D Point Clouds
</a> by Liu et al. (CVPR 2019). Code and data released in <a href="https://github.com/xingyul/flownet3d">GitHub</a>.
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>.
