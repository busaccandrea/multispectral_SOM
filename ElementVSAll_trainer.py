import os
import matplotlib.pyplot as plt
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ElementVSAll import ElementVSAll, train_model
from Pavencoder import split_data
from sklearn.preprocessing import MinMaxScaler, normalize
from torch.utils.data import random_split
from ElementVSAllDataset import ElementVSAllDataset

if __name__ == '__main__':
    """ 
        elements to recognise:
            [Ca, Ti, Mn, Fe, Cu, Zn, Se, Hg, Pb, Co]
        for the moment, Ca is excluded
    """
    for element in range(1,10):
        # define which device is available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data file name to use for training
        data = np.load('data/Edf20MS/classes/' + str(element) + '.npy')
        csvfile = 'data/Edf20MS/classes/' + str(element) + '.csv'
        
        # normalization of inputs
        data = normalize(data, axis=1, norm='max')

        # define dataset object
        el_dataset = ElementVSAllDataset(csv_file=csvfile, data=data)

        # define ratio between train set and test set
        trainset_ratio = 0.8
        trainsize = int(data.shape[0] * trainset_ratio)
        testsize = len(el_dataset) - trainsize

        # split dataset into training set and test set.
        training_set, test_set = random_split(el_dataset, [testsize, trainsize])
        print('test set length:', len(test_set), testsize, 'training set length:', len(training_set), trainsize, 'datashape', data.shape)

        # define model
        model_filename = 'data/ElementVSAll/' +  '0.p'
        if os.path.isfile(model_filename):
            print('Model and checkpoint files found. Loading.')
            model = torch.load(model_filename)
        else:
            print('Model not found. Creating...')
            input_size = data.shape[1] # size of each sample
            model = ElementVSAll(input_size=input_size)
            print('Done.')

        print('Moving model to', device)
        model.to(device)
        model.double()

        # train
        epochs = 50

        batch_size = 256
        learning_rate = 0.006
        print('Training model for', epochs, 'epochs...')
        train_model(model, training_set, test_set, epochs=epochs, batch_size=batch_size, lr=learning_rate, model_filename=str(element))
        print('\nDone.')