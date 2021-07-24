import os
import torch
import numpy as np
from ChemElementRegressor import ChemElementRegressor, train_regressor, ChemElementRegressor_Convbased
from sklearn.preprocessing import normalize
from torch.utils.data import random_split
from ChemElementRegressorDataset import ChemElementRegressorDataset, BalancedRandomSampler
import csv
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """ 
        elements to recognise:
            [Ca, Ti, Mn, Fe, Cu, Zn, Se, Hg, Pb, Co]
        for the moment, Ca is excluded
    """
    # data file name to use for training
    data = np.load('data/Edf20MS/data.npy')
    elements = [
        # 1,
        2,
        3,
        4, # index out of bound
        5, # index out of bound
        6,
        7,
        8,
        9
    ]
    for element in elements:
        print("Training of element:", element)
        # define which device is available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        csvfile = 'data/Edf20MS/classes/' + str(element) + 'regr_restricted.csv'
        
        # pre-processing of inputs
        data = normalize(data, axis=1, norm='max')

        # training parameters
        epochs = 25
        batch_size = 256
        learning_rates = [0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006] # using a list to give different learning rates for each element
        
        # define dataset object
        n_high_counts = 0
        labels = []
        row_numbers = []
        with open(csvfile, newline='') as infile: # this 'with' operation takes 0.01 sec
            csvreader = csv.reader(infile, delimiter=' ')
            for row in csvreader:
                # print('DEBUG: ROW', row)
                if float(row[-1]) > 200:
                    n_high_counts += 1
                row_numbers.append(float(row[0]))
                labels.append(float(row[-1]))
        n_low_counts = data.shape[0] - n_high_counts
        # print('DEBUG: n_low_counts:', n_low_counts, '\nDEBUG: n_high_counts:', n_high_counts)

        trainset_ratio = 0.8
        trainsize = int(data.shape[0] * trainset_ratio)
        testsize = data.shape[0] - trainsize
        xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=1-trainset_ratio, random_state=2233)

        training_set = ChemElementRegressorDataset(csv_file=csvfile, data=xtrain, labels=ytrain, row_numbers=row_numbers) #row numbers
        test_set = ChemElementRegressorDataset(csv_file=csvfile, data=xtest, labels=ytest, row_numbers=row_numbers) #row numbers
        
 
        # split dataset into training set and test set.
        print('test set length:', testsize, '\ntraining set length:', trainsize, '\ndatashape', data.shape)
        
        training_sampler = BalancedRandomSampler(training_set, high_low_ratio=0.5) # pass the sampler to train method
        test_sampler = BalancedRandomSampler(test_set, high_low_ratio=0.5) # pass the sampler to train method

        # define model
        model_filename = 'data/ElementVSAll/c' + str(element)+ 'regr.p'
        if os.path.isfile(model_filename):
            print('Model and checkpoint files found. Loading.')
            model = torch.load(model_filename)
        else:
            print('Model not found. Creating...')
            input_size = data.shape[1] # size of each sample
            model = ChemElementRegressor(input_size=input_size)
            print('Done.')

        print('Moving model to', device)
        model.to(device)
        model.double()

        # train
        print('Training model for', epochs, 'epochs...')
        train_regressor(model, training_set, test_set, epochs=epochs, batch_size=batch_size, lr=learning_rates[element], sampler={'training':training_sampler, 'test':test_sampler}, model_filename=str(element)+'conv.p')
        print('\nDone.')