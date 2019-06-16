import random
import torch
import torch.utils.data as data
import numpy as np
import gulpio


class GulpVideoDataset(data.Dataset):

    def __init__(self, data_path, samples_per_video, num_frames, step_size, tsn=1,
                transform=None,
                target_transform=None, random_offset=True):
        r"""Simple data loader for GulpIO format.
            Args:
                data_path (str): path to GulpIO dataset folder
                label_path (str): path to GulpIO label dictionary matching
            label ids to label names
                num_frames (int): number of frames to be fetched.
                step_size (int): number of frames skippid while picking
            sequence of frames from each video.
                is_va (bool): sets the necessary augmention procedure.
                transform (object): set of augmentation steps defined by
            Compose(). Default is None.
                target_transform (func): performs preprocessing on labels if
            defined. Default is None.
                stack (bool): stack frames into a numpy.array. Default is True.
                random_offset (bool): random offsetting to pick frames, if
            number of frames are more than what is necessary.
        """
        print('data loader')
        self.data_path = data_path
        self.samples_per_video = samples_per_video
        self.num_frames = num_frames
        self.step_size = step_size
        self.transform = transform
        self.target_transform = target_transform
        self.gd = gulpio.GulpDirectory(data_path)
        self.label2idx = self.gd._load_label_dict()
        self.classes = sorted(self.label2idx, key=self.label2idx.get)
        self.random_offset = random_offset
        self.make_db()
        self.tsn = tsn if tsn > 0 else 1
        print('data loading --', len(self.data))

    def __getitem__(self, index):
        """
        With the given video index, it fetches frames. This functions is called
        by Pytorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        video_id = self.data[index]['video_id']
        frame_start, frame_end = self.data[index]['frame_index']
        label = self.data[index]['label']

        target = {'video_id': video_id,
                'label': label,
                'label_name': self.classes[label]}
        if self.target_transform:
            target = self.target_transform(target)

        if self.num_frames == 1:
            if self.random_offset:
                num_frames_sample = frame_end - frame_start
                frame_indices = random.sample(range(frame_start,frame_end), min(num_frames_sample, self.tsn))
            else:
                if self.tsn == 1:
                    frame_indices = [(frame_start + frame_end - 1) // 2]
                else:
                    frame_indices = tuple(np.linspace(frame_start, frame_end-1, self.tsn).round())
            clip, _ = self.gd[video_id, frame_indices]
            # if video is shorter than necessary
            for i in range(self.tsn):
                if len(clip) >= self.tsn:
                    break
                clip.append(clip[i])
            # augmentation
            if self.transform:
                self.transform.randomize_parameters()
                clip = [self.transform(img) for img in clip]
            # format data to torch tensor
            if self.tsn == 1:
                clip = torch.from_numpy(clip[0].transpose(2,0,1))
            else:
                clip_ = []
                for c in clip:
                    clip_.append(torch.from_numpy(c.transpose(2,0,1)))
                clip = clip_
        else:
            num_frames_sample = frame_end - frame_start
            num_frames_necessary = self.num_frames * self.step_size
            diff =  num_frames_sample - num_frames_necessary
            offset = 0
            if diff > 0:
                if self.random_offset:
                    offset = random.randint(0, diff)
                else:
                    offset = diff // 2
            # set target frames to be loaded
            frames_slice = slice(offset + frame_start, offset + frame_start + num_frames_necessary,
                                self.step_size)
            clip, _ = self.gd[video_id, frames_slice]
            # if video is shorter than necessary
            for i in range(self.num_frames):
                if len(clip) >= self.num_frames:
                    break
                clip.append(clip[i])
            # augmentation
            if self.transform:
                self.transform.randomize_parameters()
                clip = [self.transform(img) for img in clip]
            # format data to torch tensor
            clip = torch.from_numpy(np.stack(clip, 0).transpose(3, 0, 1, 2))

        return clip, target

    def __len__(self):
        """
        This is called by PyTorch dataloader to decide the size of the dataset.
        """
        return len(self.data)

    def make_db(self):
        self.data = []
        num_frames_necessary = self.num_frames * self.step_size
        video_dict_item_sorted = sorted(self.gd.merged_meta_dict.items())
        for video_id, video_info in video_dict_item_sorted:
            label_idx = self.label2idx[ video_info['meta_data'][0]['label'] ]
            num_frames_video = len(video_info['frame_info'])
            if self.samples_per_video == 1:
                self.data.append( {
                    'video_id': video_id,
                    'label': label_idx,
                    'frame_index': [0, num_frames_video]} )
            elif self.samples_per_video > 1:
                num_diff = max(0, num_frames_video - num_frames_necessary)
                step = num_diff / (self.samples_per_video - 1)
                for j in range(self.samples_per_video):
                    frame_start =  min(num_frames_video-1,int(j*step))
                    frame_end = min(num_frames_video, frame_start + num_frames_necessary)
                    self.data.append( {
                        'video_id': video_id,
                        'label': label_idx,
                        'frame_index': [frame_start, frame_end]} )
            else:
                print("samples_per_video=", self.samples_per_video)
                raise

    def multiply_data(n_times):
        self.data = self.data * n_times
