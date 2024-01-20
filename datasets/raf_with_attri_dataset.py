import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import random

from PIL import Image

AU_names = ['Inner brow raiser',
            'Outer brow raiser',
            'Brow lowerer',
            'Upper lid raiser',
            'Cheek raiser',
            'Lid tightener',
            'Nose wrinkler',
            'Upper lip raiser',
            'Nasolabial deepener',
            'Lip corner puller',
            'Sharp lip puller',
            'Dimpler',
            'Lip corner depressor',
            'Lower lip depressor',
            'Chin raiser',
            'Lip pucker',
            'Tongue show',
            'Lip stretcher',
            'Lip funneler',
            'Lip tightener',
            'Lip pressor',
            'Lips part',
            'Jaw drop',
            'Mouth stretch',
            'Lip bite',
            'Nostril dilator',
            'Nostril compressor',
            'Left Inner brow raiser',
            'Right Inner brow raiser',
            'Left Outer brow raiser',
            'Right Outer brow raiser',
            'Left Brow lowerer',
            'Right Brow lowerer',
            'Left Cheek raiser',
            'Right Cheek raiser',
            'Left Upper lip raiser',
            'Right Upper lip raiser',
            'Left Nasolabial deepener',
            'Right Nasolabial deepener',
            'Left Dimpler',
            'Right Dimpler']
AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22',
          '23', '24', '25', '26', '27', '32', '38', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10',
          'L12', 'R12', 'L14', 'R14']
AU_ids_names = {
    '1': 'Inner brow raiser',
    '2': 'Outer brow raiser',
    '4': 'Brow lowerer',
    '5': 'Upper lid raiser',
    '6': 'Cheek raiser',
    '7': 'Lid tightener',
    '9': 'Nose wrinkler',
    '12': 'Lip corner puller',
    '14': 'Dimpler',
    '15': 'Lip corner depressor',
    '16': 'Lower lip depressor',
    '20': 'Lip stretcher',
    '23': 'Lip tightener',
    '26': 'Jaw drop'
}
exp_names = ['Su', 'Fe', 'Di', 'Ha', 'Sa', 'An', 'Ne', 'Co']
au_AU_map = ['1', '2', '4', '5', '6', '7', '9', '12', '14', '15', '16', '20', '23', '26']
exp_au_map = {'Su': [0, 1, 3, 13],
              'Fe': [0, 1, 2, 3, 5, 11, 13],
              'Di': [6, 9, 10],
              'Ha': [4, 7],
              'Sa': [0, 2, 9],
              'An': [2, 3, 5, 12],
              'Ne': [],
              'Co': [7, 8],
              }
label_au_map = {'0': [0, 1, 3, 13],
                '1': [0, 1, 2, 3, 5, 11, 13],
                '2': [6, 9, 10],
                '3': [4, 7],
                '4': [0, 2, 9],
                '5': [2, 3, 5, 12],
                '6': [],
                '7': [7, 8]
                }


class RAF_with_attr_dataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.img_paths = []
        if train:
            f = open(self.root + f'/EmoLabel/train.txt')
        else:
            f = open(self.root + f'/EmoLabel/test.txt')

        for line in f.readlines():
            self.img_paths.append(line.split())

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # path, exp_label, gender, race, age = self.img_paths[idx]
        path, exp_label = self.img_paths[idx][0], int(self.img_paths[idx][1]) - 1
        image = Image.open(self.root + '/Image/aligned_224/' + path.replace('_aligned', '')).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        # AUs = [0] * 14
        # for i in label_au_map[exp_label]:
        #     AUs[i] = 1
        return image, int(exp_label)
