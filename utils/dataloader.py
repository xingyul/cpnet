
import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'dataloader_utils'))

import spatial_transforms
import target_transforms
from mean import get_mean, get_std
from datasets.GulpVideoDataset import GulpVideoDataset
import cv2

def get_loader(root, train_transform, val_transform, target_transform, batch_size=64, num_frames=8, step_size=4, val_samples=1, n_threads=16, train_repeat=1, tsn=1, training=True, val=True, test=False):

    if training:
        # train dataset
        training_data = GulpVideoDataset(
            os.path.join(root, 'train'),
            1, num_frames, step_size, tsn,
            transform=train_transform,
            target_transform=target_transform,
            random_offset=True)
        if train_repeat > 1:
            training_data.multiply_data(train_repeat)
        # train loader
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_threads,
            sampler=None,
            drop_last=True)
    else:
        train_loader = None

    if val:
        # validation dataset
        validation_data = GulpVideoDataset(
            os.path.join(root, 'val'),
            val_samples, num_frames, step_size,
            transform=val_transform,
            target_transform=target_transform,
            random_offset=False)
        # val loader
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=batch_size // val_samples * val_samples,
            shuffle=False,
            num_workers=n_threads,
            drop_last=True)
    else:
        val_loader = None

    if test:
        # test dataset
        test_data = GulpVideoDataset(
            os.path.join(root, 'test'),
            val_samples, num_frames, step_size,
            transform=val_transform,
            target_transform=target_transform,
            random_offset=False)
        # test loader
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size // val_samples * val_samples,
            shuffle=False,
            num_workers=n_threads,
            drop_last=True)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader




if __name__ == '__main__':
    train_loader, val_loader = get_loader(root='/mnt/xingyliu/kinetics_subset_5_2_gulp/', batch_size=64, num_frames=8, step_size=1, target_image_size=112, crop_image_size=240, val_samples=1, n_threads=16)
    print('run')
    begin_epoch = 0
    n_epochs = 1
    for epoch in range(begin_epoch, n_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            clips = inputs.data.numpy()
            labels = targets.data.numpy()
            clip = clips[3]

            clip = ((clip - clip.min()) / (clip.max() - clip.min()) * 255).astype('uint8')
            clip = np.transpose(clip, [1,2,3,0])
            if not os.path.exists('unit_test'):
                os.system('mkdir unit_test')
            for c, f in enumerate(clip):
                cv2.imwrite(os.path.join('unit_test', 'video-{}-{}.jpg'.format(i, c)), f)
            print(labels.shape)
