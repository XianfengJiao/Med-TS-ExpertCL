import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

sys.path.append("..")
from utils.augmentation_utils import *

class Aug_Dataset(Dataset):
    def __init__(self, input_data, config, kfold=0, aug_type='strong'):
        self.aug_type = aug_type
        self.config = config

        self.x = self._get_n2n_data(input_data) if config.n2n else input_data
        self.lens = [len(i) for i in self.x]
        self.configure_transform()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        lens = self.lens[index]
        return x, lens

    def _get_n2n_data(self, x):
        length = len(x)
        x_n2n = []
        for i in range(length):
            for j in range(len(x[i])):
                x_n2n.append(np.array(x[i])[:j+1])
        return x_n2n

    def configure_transform(self):
        if self.aug_type == 'strong':
            self.data_transform = data_transform_strong
        elif self.aug_type == 'week':
            self.data_transform = data_transform_week
        elif self.aug_type == 'both':
            self.data_transform = data_transform_both
        else:
            raise Exception("Invalid augmentation type:", self.aug_type)

    def collate_fn(self, dataset):
        x, lens = zip(*dataset)
        if len(np.array(x[0]).shape) == 1:
            x_pad = torch.zeros(len(x), max(lens), 1).float()
        else: 
            x_pad = torch.zeros(len(x), max(lens), len(x[0][0])).float()
        for i, xx in enumerate(x):
            end = lens[i]
            x_pad[i,:end] = torch.FloatTensor(np.array(xx)).unsqueeze(1) if len(np.array(xx).shape) == 1 else torch.FloatTensor(np.array(xx))

        aug_x1, aug_x2 = self.data_transform(x_pad.permute(0, 2, 1), self.config)

        aug_x1 = aug_x1.float() if type(aug_x1) == torch.Tensor else torch.FloatTensor(aug_x1)
        aug_x2 = aug_x2.float() if type(aug_x2) == torch.Tensor else torch.FloatTensor(aug_x2)

        return x_pad, aug_x1.permute(0, 2, 1), aug_x2.permute(0, 2, 1), torch.LongTensor(lens)
    
        
        