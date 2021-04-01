from __future__ import print_function, division
from functools import total_ordering
import os
from re import S
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from time import time


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, pbl = sample['rgb'], sample['pb']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        rgb = rgb.transpose((2, 0, 1))
        return {'image': torch.from_numpy(rgb),
                'landmarks': torch.from_numpy(pbl)}


# plt.ion() # modalità interattiva
class CustomDataset(Dataset):
    """ Build a custom dataset for RGB2Gray model """
    def __init__(self, csv_file, root_dir, transform=ToTensor()):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. row example: "RGB_0.png,PbL0.png"
            root_dir (string): Directory with all the images.
            deleted ------ transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.trasform = transform # not used for now.
    
    # mandatory override of methods __len__ and __getitem__
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # row example: "RGB_0.png,PbL0.png"
        img_name = self.csv_file.iloc[index, 0]
        print(img_name)
        RGB_i = io.imread(img_name)
        PbL_i = self.csv_file.iloc[index, 1]

        # return images in a dict
        sample = {'rgb': RGB_i, 'pb': PbL_i}

        return sample