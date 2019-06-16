======
GulpIO
======

About
=====

Binary storage format for deep learning on videos.

Status
======

.. image:: https://travis-ci.com/TwentyBN/GulpIO.svg?token=S1oNqvRREH7GP7VjDu9s&branch=master
    :target: https://travis-ci.com/TwentyBN/GulpIO

Install
=======

There are many ways to setup a Python environment for installation. Here we
outline an approach using *virtualenvironment*. Note: this package does not
support legacy Python and will only work with Python 3.x.

The following will setup a virtualenvironment and activate it:

.. code::

    python -m venv gulpio
    source gulpio/bin/activate

Then, install the package using:

.. code::

    pip install gulpio

Usage
=====

'Gulp' a Dataset
----------------

The ``gulpio`` package has been designed to be infinitely hackable and to support
arbitrary datasets. To this end, we are providing a so-called *adapter
pattern*. Specifically there exists an abstract class in ``gulpio.adapters``:
the ``AbstractDatasetAdapter``.  In order to ingest your dataset, you basically
need to implement your own custom adapter that inherits from this.

You should be able to get going quickly by looking at the following examples,
that we use internally to gulp our video datasets.

* The class: ``gulpio.adapters.Custom20BNJsonAdapter`` `adapters.py <src/main/python/gulpio/adapters.py>`_
* The script: ``gulp_20bn_json_videos`` `command line script <src/main/scripts/gulp_20bn_json_videos>`_

And an example invocation would be:

.. code::

   gulp_20bn_json_videos videos.json input_dir output_dir
   # ...

Additionally, if you would like to ingest your dataset from the command line,
the ``register_adapter`` script can be used to generate the command line interface
for the new adapter. Write your adapter that inherits from the ``AbstractDatasetAdapter``
in the ``adapter.py`` file, then simply call:

.. code::

    gulp_register_adapter gulpio.adapters <NewAdapterClassName>

The script that provides the command line interface will be in the main directory of the repository. To use it, execute ``./new_adapter_class_name``.


Sanity check the 'Gulped' Files
-------------------------------

A very basic test to check the correctness of the gulped files is provided by the ``gulp_sanity_check`` script.
For execution run:

.. code::

    gulp_sanity_check <folder containing the gulped files> 

It tests:

* The presence of any content in the ``.gulp`` and ``.gmeta``-files
* The file size of the ``.gulp`` file corresponds to the required file size that is given in the ``.gmeta`` file
* Duplicate appearances of any video-ids

The file names of the files where any test fails will be printed. Currently no script to fix possible errors is
provided, 'regulping' is the only solution.


Read a 'Gulped' Dataset
-----------------------

In order to read from the gulps, you can let yourself be inspired by the
following snippet:

.. code:: python

    # import the main interface for reading
    from gulpio import GulpDirectory
    # instantiate the GulpDirectory
    gulp_directory = GulpDirectory('/tmp/something_something_gulps')
    # iterate over all chunks
    for chunk in gulp_directory:
        # for each 'video' get the metadata and all frames
        for frames, meta in chunk:
            # do something with the metadata
            for i, f in enumerate(frames):
                # do something with the frames
                pass

Alternatively, a video with a specific ``id`` can be directly accessed via:

.. code:: python

    # import the main interface for reading
    from gulpio import GulpDirectory
    #instantiate the GulpDirectory
    gulp_directory = GulpDirectory('/tmp/something_something_gulps')
    frames, meta = gulp_directory[<id>]

For down-sampling or loading only a part of a video, a python slice can be
passed as well:

.. code:: python

    frames, meta = gulp_directory[<id>, slice(1,10,2)]

or:

.. code:: python

    frames, meta = gulp_directory[<id>, 1:10:2]


Loading Data
------------

You can use GulpIO data iterator and augmentation functions to load GulpIO dataset into memory.
For a working example given on different deep learning libraries please refer to  ``examples/GulpIOTrainingExample.ipynb``. 

We provide archetypical ``dataset`` wrappers that work for general supervised cases of image and video datasets. If you need a particular use,
you might need to create your own dataset by inheriting ``dataset.py`` and overwriting ``__getitem__`` and ``__len__``. 

Below is an example loading an image dataset with GulpIO loader and defining augmentation pipeline. 
Transformations are applied to each instance on the fly. Some transformations have separate video and image versions since some of the augmentations
need to be aligned video-wise. 

.. code:: python 

    from gulpio.dataset import GulpImageDataset
    from gulpio.loader import DataLoader
    from gulpio.transforms import Scale, CenterCrop, Compose, UnitNorm

    # define data augmentations. Notice that there are different functions for videos and images
    transforms = Compose([
                          Scale(28),  # resize image by the shortest edge
                          CenterCrop(28),
                          UnitNorm(),  # instance wise mean and std norm
                        ])

    # define dataset wrapper and pick this up by the data loader interface.
    dataset = GulpImageDataset('/path/to/train_data', transform=transforms)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    dataset_val = GulpImageDataset('/path/to/validation_data/', transform=transforms)
    loader_val = DataLoader(dataset_val, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

Here we iterate through the dataset we loaded. Iterator returns data and label as numpy arrays. You might need to cast these into the format of you
deep learning library.

.. code:: python

    for data, label in loader:
        # train your model here
        # ...



Format Description
==================

When gulping a dataset, two different files are created for every chunk: a
``*.gulp`` data file that contains the actual data and a ``*.gmeta`` meta file
that contains the metadata.

The layout of the ``*.gulp`` file is as follows:

.. code::

    |-jpeg-|-pad-|-jpeg-|-pad-|...


Essentially, the data file is simply a series of concatenated JPEG images, i.e.
the frames of the video. Each frame is padded to be divisible by four bytes,
since this makes it easier to read JPEGs from disk.

Here is a more visual example:

.. image:: docs/data_file_layout.png

As you can see there are 6 *records* in the example. They have the following
paddings and lengths:

=====  =====  =====
FRAME  LEN    PAD
=====  =====  =====
0      4      1
1      4      2
2      4      0
3      4      1
4      4      3
5      8      1
=====  =====  =====

The layout of the meta file is a mapping, where each ``id`` representing a
video is mapped to two further mappings, ``meta_data``, which contains
arbitrary, user-defined meta-data. And a triplet, ``frame_info``, which
contains the offset (index) into the data file, the number of bytes used for
padding and the total length of the frame (including padding). (``[<offset>,
<padding>, <total_length>]``.) The `frame_info` is required to recover the
frames from the data file.

.. code::

    'id'
      |
      |-> meta_data: [{}]
      |
      |-> frame_info: [[], [], ...]
    .
    .
    .


By default, the meta file is serialized in JSON format.

For example, here is a meta file snippet:

.. code::

    {"702766": {"frame_info": [[0, 3, 7260],
                               [7260, 3, 7252],
                               [14512, 2, 7256],
                               [21768, 2, 7260],
                               [29028, 1, 7308],
                               [36336, 1, 7344],
                               [43680, 0, 7352],
                               [51032, 1, 7364],
                               [58396, 0, 7348],
                               [65744, 1, 7352],
                               [73096, 1, 7352],
                               [80448, 1, 7408],
                               [87856, 1, 7400],
                               [95256, 0, 7376],
                               [102632, 1, 7384],
                               [110016, 2, 7404],
                               [117420, 0, 7396],
                               [124816, 1, 7400],
                               [132216, 2, 7428],
                               [139644, 1, 7420],
                               [147064, 0, 7428],
                               [154492, 2, 7472],
                               [161964, 3, 7456],
                               [169420, 2, 7444],
                               [176864, 2, 7436]],
                "meta_data":  [{"label": "something something",
                                "id":    702766}]},
     "803959": {"frame_info": [[184300, 1, 9256],
                               [193556, 3, 9232],
                               [202788, 2, 9340],
                               [212128, 2, 9184],
                               [221312, 1, 9112],
                               [230424, 3, 9100],
                               [239524, 0, 9144],
                               [248668, 1, 9120],
                               [257788, 0, 9104],
                               [266892, 0, 9220],
                               [276112, 1, 9140],
                               [285252, 1, 9076],
                               [294328, 2, 9100],
                               [303428, 0, 9224],
                               [312652, 3, 9200],
                               [321852, 3, 9136],
                               [330988, 2, 9136],
                               [340124, 1, 9152],
                               [349276, 0, 8984],
                               [358260, 1, 9048],
                               [367308, 0, 9116],
                               [376424, 1, 9136],
                               [385560, 1, 9108],
                               [394668, 2, 9084],
                               [403752, 1, 9112],
                               [412864, 2, 9108]],
                "meta_data":  [{"label": "something something",
                                "id":    803959}]},
     "803957": {"frame_info": [[421972, 2, 8592],
                               [430564, 1, 8608],
                               [439172, 2, 8872],
                               [448044, 3, 8852],
                               [456896, 2, 8860],
                               [465756, 0, 8908],
                               [474664, 2, 8912],
                               [483576, 1, 8884],
                               [492460, 1, 8752],
                               [501212, 3, 8692],
                               [509904, 0, 8612],
                               [518516, 0, 8816],
                               [527332, 2, 8784],
                               [536116, 1, 8840],
                               [544956, 1, 8844],
                               [553800, 1, 8988],
                               [562788, 0, 8992],
                               [571780, 0, 8972],
                               [580752, 3, 9044],
                               [589796, 2, 9012],
                               [598808, 3, 9060],
                               [607868, 2, 9032],
                               [616900, 1, 9052],
                               [625952, 2, 9056],
                               [635008, 0, 9084],
                               [644092, 2, 9100]],
                "meta_data":  [{"label": "something something",
                                "id":    803957}]},
     "773430": {"frame_info": [[653192, 1, 7964],
                               [661156, 2, 7996],
                               [669152, 1, 7960],
                               [677112, 0, 8024],
                               [685136, 0, 8008],
                               [693144, 1, 7972],
                               [701116, 0, 7980],
                               [709096, 0, 8036],
                               [717132, 0, 8016],
                               [725148, 0, 8016],
                               [733164, 1, 8004],
                               [741168, 1, 8008],
                               [749176, 1, 7996],
                               [757172, 1, 8016],
                               [765188, 1, 8032],
                               [773220, 0, 8040],
                               [781260, 2, 8044],
                               [789304, 2, 8004],
                               [797308, 1, 8008],
                               [805316, 0, 8056],
                               [813372, 3, 8088],
                               [821460, 0, 8044]],
                "meta_data":  [{"label": "something something",
                                "id":    773430}]},
     "803963": {"frame_info": [[829504, 2, 8952],
                               [838456, 1, 8928],
                               [847384, 0, 8972],
                               [856356, 1, 8992],
                               [865348, 1, 8936],
                               [874284, 1, 8992],
                               [883276, 3, 8988],
                               [892264, 1, 9008],
                               [901272, 2, 8996],
                               [910268, 2, 8976],
                               [919244, 0, 9180],
                               [928424, 0, 9128],
                               [937552, 2, 9100],
                               [946652, 2, 9096],
                               [955748, 3, 9044],
                               [964792, 0, 9096],
                               [973888, 2, 9068],
                               [982956, 1, 8996],
                               [991952, 3, 8928],
                               [1000880, 1, 9040],
                               [1009920, 0, 9084],
                               [1019004, 0, 9076],
                               [1028080, 2, 9056],
                               [1037136, 2, 9040],
                               [1046176, 2, 9052],
                               [1055228, 3, 9096]],
                "meta_data":  [{"label": "something something",
                                "id":    803963}]}
    }

Benchmarks
==========

* Benchmarks are available in a seperate repo: https://github.com/TwentyBN/GulpIO-benchmarks

Prior Art
=========

* Inspired by: MXNet based RecordIO: http://mxnet.io/architecture/note_data_loading.html
* GulpIO data loader is branched from great `PyTorch <http://pytorch.org>`_ implementation.

License
=======

All code except ``gulpio.loader`` is Copyright (c) Twenty Billion Neurons and
licensed under the MIT License, see the file ``LICENSE.txt`` for details.

The code in ``gulpio.loader`` which came from the PyTorch project is licensed
under a 3-clause BSD License, see the file ``LICENSE_PYTORCH`` for details.
