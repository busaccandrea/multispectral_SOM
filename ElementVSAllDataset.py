from functools import total_ordering
import os
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from time import time


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        row, class_ = sample['row'], sample['class']

        return {'row': torch.from_numpy(row),
                'class': torch.from_numpy(class_)}


# plt.ion() # modalità interattiva
class ElementVSAllDataset(Dataset):
    """ Build a custom dataset for RGB2Gray model """
    def __init__(self, csv_file, data, transform=ToTensor()):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. row example: "5000,0". before comma = #row_of_data, afret comma=class
            data (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file, sep=' ')
        self.data = data
        self.trasform = transform
    
    # mandatory override of methods __len__ and __getitem__
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # row example: "RGB_0.png,PbL0.png"
        row_number = self.csv_file.iloc[index, 0]
        row = self.data[int(row_number)]
        cl = self.csv_file.iloc[index, 1]
        class_value = int(cl)
        class_ = torch.tensor([1 - class_value, class_value])

        # return images in a dict
        return {'row': row, 'class': class_}
    
    def get_subset(self, indices:list):
        return Subset(self, indices)