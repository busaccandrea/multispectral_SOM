import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, Subset
from time import time
import numpy as np
from torch.utils.data.sampler import Sampler
import torchtest


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        row, class_ = sample['row'], sample['class']

        return {'row': torch.from_numpy(row),
                'class': torch.from_numpy(class_)}


class ChemElementRegressorDataset(Dataset):
    """ Build a custom dataset for ChemElementRegressor model """
    def __init__(self, csv_file, data, labels, row_numbers, transform=ToTensor()):
        # print('DEBUG: creo il dataset')
        self.csv_file = pd.read_csv(csv_file, sep=' ')
        self.data = data
        self.labels = np.array(labels)
        self.row_numbers = np.array(row_numbers)
        self.high_count_idxs = np.where(self.labels>200)[0] # where label is > 200
        self.low_count_idxs = np.where(self.labels<=200)[0] # where label is <= 200
        self.trasform = transform
        self.pick_high = True
        # print('DEBUG: dataset creato.')
        
    # mandatory override of methods __len__ and __getitem__
    def __len__(self):
        return self.data.shape[0]


    def __getitem__concsv(self, index):
        # print('\n========== inizio getitem in ChemElementRegressorDataset ==========')
        if torch.is_tensor(index):
            index = index.tolist()
        # print('DEBUG: CHOSEN INDEX:', index)

        row_number = self.csv_file.iloc[index-1, 0]
        # print('DEBUG: row number:', row_number)
        row = self.data[int(row_number)]
        # print('DEBUG: self.data[int(row_number)] :', self.data[int(row_number)])
        
        # row example: "RGB_0.png,PbL0.png"
        val = self.csv_file.iloc[index-1, 1]
        value = int(val)
        # print('DEBUG: value:', value)
        value = torch.tensor([value])
        # # print('========== fine __getitem__ in ChemElementRegressorDataset.py ==========\n')
        return {'row': row, 'counts': value}


    def __getitem__Llll(self, index):
        # # print('\n========== inizio getitem in ChemElementRegressorDataset ==========')
        if torch.is_tensor(index):
            index = index.tolist()
        # print('DEBUG: CHOSEN INDEX:', index)

        # row_number = self.data[index-1]
        # # print('DEBUG: row number:', row_number)
        row = self.data[index-1]
        # print('DEBUG: self.data[index-1] :', self.data[index-1])
        
        # row example: "RGB_0.png,PbL0.png"
        val = self.labels[index-1]
        value = int(val)
        # print('DEBUG: value:', value)
        value = torch.tensor([value])
        # # print('========== fine __getitem__ in ChemElementRegressorDataset.py ==========\n')
        return {'row': row, 'counts': value}


    def __getitemMIO__(self, index):
        # # print('==========inizio __getitem__ in ChemElementRegressorDataset.py==========')
        # print('DEBUG: get item index', index)
        if torch.is_tensor(index):
            index = index.tolist()
        else:
            index = [index]
        
        # row example: "1000,5.0"
        val = self.csv_file.iloc[index, 1]
        value = int(val)
        # print('DEBUG: val', value)
        if self.pick_high:
            # value must be greater then 200.
            if value >= 200:
                self.pick_high = False
            else:
                # choose randomly between self.high count indexes
                index = np.random.choice(self.high_count_idxs, 1)
        else:
            # value must be lower then 200.
            if value < 200:
                self.pick_high = True
            else:
                # choose randomly between self.low count indexes
                index = np.random.choice(self.low_count_idxs, 1)

        # print(index, end='\r\n')

        val = self.csv_file.iloc[index, 1]
        value = int(val)
        # print('DEBUG: FINAL index', index)
        # print('DEBUG: FINAL value', value)
        
        row_number = self.csv_file.iloc[index, 0]
        row = self.data[int(row_number)] 
        value = torch.tensor([value])

        # # print('==========fine __getitem__ in ChemElementRegressorDataset.py==========')
        return {'row': row, 'counts': value}

    def __getitem__(self, index):
        # # print('==========inizio __getitem__ in ChemElementRegressorDataset.py==========')
        if torch.is_tensor(index):
            index = index.tolist()
        else:
            index = [index]
        
        # print('DEBUG: get item index', index)
        # row example: "1000,5.0"
        val = self.labels[index]
        value = int(val)
        # print('DEBUG: val', value)
        if self.pick_high:
            # value must be greater then 200.
            if value >= 200:
                self.pick_high = False
            else:
                # choose randomly between self.high count indexes
                index = np.random.choice(self.high_count_idxs, 1)
        else:
            # value must be lower then 200.
            if value < 200:
                self.pick_high = True
            else:
                # choose randomly between self.low count indexes
                index = np.random.choice(self.low_count_idxs, 1)

        val = self.labels[index]
        value = int(val)
        
        row = self.data[index]
        value = torch.tensor([value])

        # # print('==========fine __getitem__ in ChemElementRegressorDataset.py==========')
        return {'row': row, 'counts': value}

    # def get_subset(self, indices:list):
    #     return Subset(self, indices)

                
class BalancedRandomSampler(Sampler):
    def __init__(self, data_set:ChemElementRegressorDataset, high_low_ratio=0.5):
        # # print('\n==========inizio __init__ in balancedrandomsampler.py==========')
        self.data_set = data_set
        self.dataset_len = len(self.data_set)
        # print('DEBUG: len data_set', self.dataset_len, '\nlabels', data_set.labels)

        self.ratio = high_low_ratio
        
        self.high_count_idxs = self.data_set.high_count_idxs # where label is > 200
        self.low_count_idxs = self.data_set.low_count_idxs # where label is <= 200
        # print('DEBUG: low_count_idxs, high_count_idxs:', self.low_count_idxs, self.high_count_idxs)
        # # print('==========fine __init__ in balancedrandomsampler.py==========\n')

    def __iter__(self):
        # # # print('\n========== inizio __iter__ in balancedrandomsampler.py===============')
        # print('DEBUGGGGGGGGGGGGGGGGGGGGGGGGGGG: low_count_idxs, high_count_idxs:', self.low_count_idxs, self.high_count_idxs)
        highspectra_choice = np.random.choice(self.high_count_idxs, int(self.dataset_len * self.ratio)) # pick from high_counts_idxs dataset_len/2 elements
        # # print('DEBUG: high_count choice:', highspectra_choice)
        lowspectra_choice = np.random.choice(self.low_count_idxs, int(self.dataset_len * (1 - self.ratio)))
        # # print('DEBUG: low_count choice:', type(lowspectra_choice))

        # highspectra = self.dataset.labels[highspectra_choice]
        # lowspectra = self.dataset.labels[lowspectra_choice]
        highspectra = highspectra_choice
        lowspectra = lowspectra_choice
        # # print('DEBUG: highspectra:', highspectra)
        # # print('DEBUG: lowspectra:', lowspectra)

        idxs = np.hstack([highspectra, lowspectra])
        np.random.shuffle(idxs)
        idxs = idxs.astype(int)
        # print('DEBUG: chosen ROWS by balanced randomsampler:', idxs) #this is the whole balanceddataset
        idxs = iter(idxs[:self.dataset_len])
        # # # print('========== fine __iter__ in balancedrandomsampler.py ===============\n')

        return iter(idxs)

    def __len__(self):
        return self.dataset_len