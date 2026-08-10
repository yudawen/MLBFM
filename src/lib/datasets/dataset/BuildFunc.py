from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import torch.utils.data as data


class BF(data.Dataset):
    num_classes=10
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(BF, self).__init__()
        self.split = split



        train_path=r'D:\code_and_experiment_2025_2\experiment4\BUFFv2\train_data\trainval'
        val_path=r'D:\code_and_experiment_2025_2\experiment4\BUFFv2\train_data\test'

        if split=='train':
            self.data_root=train_path+'/'

            self.img_paths = list(sorted(os.listdir(os.path.join(train_path, "image"))))
            self.gt_txt_paths = list(sorted(os.listdir(os.path.join(train_path, "gt_txt"))))

        else:
            self.data_root=val_path+'/'


            self.img_paths = list(sorted(os.listdir(os.path.join(val_path, "image"))))
            self.gt_txt_paths = list(sorted(os.listdir(os.path.join(val_path, "gt_txt"))))




        self.opt = opt
    def __len__(self):
        return len(self.img_paths)





