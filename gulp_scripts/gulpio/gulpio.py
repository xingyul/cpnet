#!/usr/bin/env python

import os
import re
import cv2
import json
import csv
import glob
import numpy as np

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from collections import namedtuple, OrderedDict
from tqdm import tqdm
from gulpio.gulpio_utils import find_files_in_subfolders, is_chunk_valid


ImgInfo = namedtuple('ImgInfo', ['loc',
                                 'pad',
                                 'length'])

EXT_DATA = '.gulp'
EXT_META = '.gmeta'


########################
########################


class AbstractDatasetAdapter(ABC):  # pragma: no cover
    """ Base class adapter for gulping (video) datasets.

    Inherit from this class and implement the `iter_data` method. This method
    should iterate over your entire dataset and for each element return a
    dictionary with the following fields:

        id     : a unique(?) ID for the element.
        frames : a list of frames (PIL images, numpy arrays..)
        meta   : a dictionary with arbitrary metadata (labels, start_time...)

    For examples, see the custom adapters below.

    """

    @abstractmethod
    def get_data(self, index=None):
        return NotImplementedError

    @abstractmethod
    def __len__(self):
        return NotImplementedError


########################
########################


class AbstractSerializer(ABC):  # pragma: no cover

    @abstractmethod
    def load(self, file_name):
        pass

    @abstractmethod
    def dump(self, thing, file_name):
        pass

class JSONSerializer(AbstractSerializer):

    def load(self, file_name):
        with open(file_name, 'r') as file_pointer:
            return json.load(file_pointer, object_pairs_hook=OrderedDict)

    def dump(self, thing, file_name):
        with open(file_name, 'w') as file_pointer:
            json.dump(thing, file_pointer)

json_serializer = JSONSerializer()


########################
########################


class GulpDirectory(object):
    """ Represents a directory containing *.gulp and *.gmeta files.

    Parameters
    ----------
    output_dir: (str)
        Path to the directory containing the files.

    Attributes
    ----------
    chunk_lookup: (dict: str -> int)
        Mapping element id to chunk index.
    merged_meta_dict: (dict: id -> meta dict)
        all meta dicts merged

    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        data_paths = sorted(find_files_in_subfolders(self.output_dir, [EXT_DATA]))

        self.valid_chunk_paths = []
        for data_path in data_paths:
            chunk_path = data_path[:-len(EXT_DATA)]
            if os.path.exists(chunk_path + EXT_META):
                self.valid_chunk_paths.append(chunk_path)
        print('# of valid gulp chuks:', len(self.valid_chunk_paths))

        self._all_chunks = [GulpChunk(chunk_path) for chunk_path in sorted(self.valid_chunk_paths)]
        self._chunk_lookup = {}
        self.merged_meta_dict = {}
        for idx, gulp_chunk in enumerate(self._all_chunks):
            for id_ in gulp_chunk.meta_dict:
                assert id_ not in self.merged_meta_dict,\
                    "Duplicate id detected {}".format(id_)
                self._chunk_lookup[id_] = idx
            self.merged_meta_dict.update(gulp_chunk.meta_dict)
        print('# of valid videos:', len(self.merged_meta_dict))

    def __getitem__(self, element):
        id_, slice_ = element
        gulp_chunk = self._all_chunks[self._chunk_lookup[id_]]
        with gulp_chunk.open():
            return gulp_chunk.read_frames(id_, slice_)

    def _load_label_dict(self):
        return json.load(open(os.path.join(self.output_dir, 'label2idx.json'), 'rb'))


class GulpChunk(object):
    """ Represents a gulp chunk on disk.

    Parameters
    ----------
    data_file_path: (str)
        Path to the *.gulp file.
    meta_file_path: (str)
        Path to the *.gmeta file.
    serializer: (subclass of AbstractSerializer)
        The type of serializer to use.

    """

    def __init__(self, chunk_path,
                 serializer=json_serializer):
        self.serializer = serializer
        self.data_file_path = chunk_path + EXT_DATA
        self.meta_file_path = chunk_path + EXT_META
        self.meta_dict = self._get_or_create_dict()
        self.fp = None

    def reset_meta(self):
        self.meta_dict = OrderedDict()

    def _get_frame_infos(self, id_):
        id_ = str(id_)
        if id_ in self.meta_dict:
            return ([ImgInfo(*info)
                     for info in self.meta_dict[id_]['frame_info']],
                    dict(self.meta_dict[id_]['meta_data'][0]))

    def _get_or_create_dict(self):
        if os.path.exists(self.meta_file_path):
            try:
                return self.serializer.load(self.meta_file_path)
            except:
                print("Failed to load the metadata:", self.meta_file_path)
                return OrderedDict()
        else:
            return OrderedDict()

    @staticmethod
    def _default_factory():
        return OrderedDict([('frame_info', []), ('meta_data', [])])

    @staticmethod
    def _pad_image(number):
        return (4 - (number % 4)) % 4

    def _append_meta(self, id_, meta_data):
        id_ = str(id_)
        if id_ not in self.meta_dict:  # implements an OrderedDefaultDict
            self.meta_dict[id_] = self._default_factory()
        self.meta_dict[id_]['meta_data'].append(meta_data)

    def _write_frame(self, id_, image):
        loc = self.fp.tell()
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        assert len(img_str) > 0
        pad = self._pad_image(len(img_str))
        record = img_str.ljust(len(img_str) + pad, b'\0')
        assert len(record) > 0
        img_info = ImgInfo(loc=loc,
                           length=len(record),
                           pad=pad)
        id_ = str(id_)
        if id_ not in self.meta_dict:  # implements an OrderedDefaultDict
            self.meta_dict[id_] = self._default_factory()
        self.meta_dict[id_]['frame_info'].append(img_info)
        self.fp.write(record)

    def _write_frames(self, id_, frames):
        for frame in frames:
            self._write_frame(id_, frame)

    @contextmanager
    def open(self, flag='rb'):
        """Open the gulp chunk for reading.

        Parameters
        ----------
        flag: (str)
            'rb': Read binary
            'wb': Write binary
            'ab': Append to binary

        Notes
        -----
        Works as a context manager but returns None.

        """
        if flag in ['wb', 'ab']:
            dirname = os.path.dirname(self.data_file_path)
            os.makedirs(dirname, exist_ok=True)
            self.fp = open(self.data_file_path, flag)
        elif flag in ['rb']:
            self.fp = open(self.data_file_path, flag)
        else:
            m = "This file does not support the mode: '{}'".format(flag)
            raise NotImplementedError(m)
        yield
        if flag in ['wb', 'ab']:
            self.flush()
        self.fp.close()

    def flush(self):
        """Flush all buffers and write the meta file."""
        self.fp.flush()
        self.serializer.dump(self.meta_dict, self.meta_file_path)

    def append(self, id_, meta_data, frames):
        """ Append an item to the gulp.

        Parameters
        ----------
        id_ : (str)
            The ID of the item
        meta_data: (dict)
            The meta-data associated with the item.
        frames: (list of numpy arrays)
            The frames of the item as a list of numpy dictionaries consisting
            of image pixel values.

        """
        self._write_frames(id_, frames)
        self._append_meta(id_, meta_data)

    def read_frames(self, id_, slice_=None):
        """ Read frames for a single item.

        Parameters
        ----------
        id_: (str)
            The ID of the item
        slice_: (slice:
            A slice with which to select frames.
        Returns
        -------
        frames (int), meta(dict)
            The frames of the item as a list of numpy arrays consisting of
            image pixel values. And the metadata.

        """
        frame_infos, meta_data = self._get_frame_infos(id_)
        frames = []
        slice_element = slice_ or slice(0, len(frame_infos))

        def extract_frame(frame_info):
            self.fp.seek(frame_info.loc)
            record = self.fp.read(frame_info.length)
            img_str = record[:len(record)-frame_info.pad]
            nparr = np.fromstring(img_str, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
            # if img.ndim > 2:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        if isinstance(slice_element, slice):
            frames = [extract_frame(frame_info)
                    for frame_info in frame_infos[slice_element]]
        else:
            frames = [extract_frame(frame_infos[element])
                    for element in slice_element if len(frame_infos) > element]
        return frames, meta_data


class ChunkWriter(object):
    """Can write from an adapter to a gulp chunk.

    Parameters
    ----------
    adapter: (subclass of AbstractDatasetAdapter)
       The adapter to get items from.

    """

    def __init__(self, output_path, adapter, skip_exist):
        self.output_path = output_path
        self.adapter = adapter
        self.skip_exist = skip_exist

    def write_chunk(self, index):
        """Write from an input slice in the adapter to an output chunk.

        Parameters
        ----------
        index: int
           The number to use from the adapter.

        """
        chunk_path = self.adapter.get_chunk_path(index)
        output_chunk = GulpChunk(os.path.join(self.output_path, chunk_path))
        if self.skip_exist:
            num_frames = is_chunk_valid(output_chunk)
            if num_frames > 0:
                return (chunk_path, 1, str(num_frames))
        data = self.adapter.get_data(index)
        id_ = data['id']
        meta_data = data['meta']
        frames = data['frames']
        if len(frames) > 0:
            output_chunk.reset_meta()
            with output_chunk.open('wb'):
                output_chunk.append(id_, meta_data, frames)
            return (chunk_path, 1, str(len(frames)))
        else:
            #print("Failed to write video with id: {}; no frames".format(id_))
            return (chunk_path, 0, str(len(frames)))


class GulpIngestor(object):
    """Ingest items from an adapter into an gulp chunks.

    Parameters
    ----------
    adapter: (subclass of AbstractDatasetAdapter)
        The adapter to ingest from.
    output_path: (str)
        The folder/directory to write to.
    num_workers: (int)
        The level of parallelism.

    """
    def __init__(self, adapter, output_path, num_workers, skip_exist=True):
        assert int(num_workers) > 0
        self.adapter = adapter
        self.output_path = output_path
        self.num_workers = int(num_workers)
        self.log_path = os.path.join(self.output_path, 'gulp_log.csv')
        self.skip_exist = skip_exist

    def __call__(self):
        os.makedirs(self.output_path, exist_ok=True)
        num_videos = len(self.adapter)
        chunk_size = min(500,max(1,num_videos//(10*self.num_workers)))
        print("Number of Videos:", num_videos)
        chunk_writer = ChunkWriter(self.output_path, self.adapter, self.skip_exist)
        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor, \
                open(self.log_path, 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                result = executor.map(chunk_writer.write_chunk, range(num_videos), chunksize=chunk_size)
                for status in tqdm(result,
                                desc='Chunks finished',
                                dynamic_ncols=True,
                                total=num_videos):
                    csv_writer.writerow(status)
        else:
            with open(self.log_path, 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for index in tqdm(range(num_videos),
                                desc='Chunks finished',
                                dynamic_ncols=True,
                                total=num_videos):
                    status = chunk_writer.write_chunk(index)
                    csv_writer.writerow(status)
