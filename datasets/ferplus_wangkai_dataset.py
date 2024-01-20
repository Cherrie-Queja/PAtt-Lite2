import torch.utils.data as data
import cv2
import numpy as np
import os
import random

from PIL import Image


class FERPlus_wangkai_dataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.img_paths = []
        if train:
            self.subset = 'train+val'
            f = open(self.root + f'/wangkai_dir/dlib_ferplus_train_center_crop_range_list_wangkai-7cls.txt')
        else:
            self.subset = 'test'
            f = open(self.root + f'/wangkai_dir/dlib_ferplus_val_center_crop_range_list_wangkai-7cls.txt')

        for line in f.readlines():
            self.img_paths.append(line.split()[0].split('/'))  # ['0','fer00001']

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        exp_label, img_name = self.img_paths[idx]
        image = Image.open(self.root + f'/wangkai_dir_MTCNN/{self.subset}/' + img_name + '.png').convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, int(exp_label)

#
# f7 = open('/home/panr/all_datasets/ferplus/wangkai_dir/dlib_ferplus_val_center_crop_range_list_wangkai-7cls.txt','a')
# f8 = open('/home/panr/all_datasets/ferplus/wangkai_dir/dlib_ferplus_val_center_crop_range_list_wangkai-8cls.txt','r')
#
# for line in f8.readlines():
#     if line.startswith('7/'):
#         continue
#     f7.write(line)
