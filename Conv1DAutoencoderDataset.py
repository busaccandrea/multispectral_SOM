from __future__ import print_function, division
from functools import total_ordering
import os
from re import S
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from time import time


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, spectra):
        return {'spectra': torch.from_numpy(spectra)}


# plt.ion() # modalità interattiva
class Conv1DAutoencoderDataset(Dataset):
    """ Build a custom dataset for Conv1DAutoencoder model """
    def __init__(self, csv_file, root_dir, transform=ToTensor()):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. row example: "RGB_0.png,PbL0.png"
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.trasform = transform
    
    # mandatory override of methods __len__ and __getitem__
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # row example: "RGB_0.png,PbL0.png"
        img_name = self.csv_file.iloc[index, 0]
        RGB_i = io.imread(img_name)
        img_name = self.csv_file.iloc[index, 1]
        PbL_i = io.imread(img_name)

        # return images in a dict
        return {'rgb': RGB_i, 'pbl': PbL_i}
    
    def get_subset(self, indices:list):
        return Subset(self, indices)