import os
import matplotlib.pyplot as plt
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from Pavencoder import Pavencoder, load_model_for_inference
from Pavencoder import split_data_from_numpy
from Pavencoder import train
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':

    # define which device is available.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # define batch size for AE
    batch_size = 256

    # data file name to use for training
    # data_filename = 'data/Edf20MS/Edf/data.npy'
    data_filename = 'data/Edf20MS/dataset.npy'
    
    # split data in two 2-dimensional tensors: 70% training set, 30% test set.
    # [training_set, test_set] = split_data_from_numpy(data_filename)
    training_set = np.load('data/Edf20MS/train/train_custom.npy')
    test_set=np.load('data/Edf20MS/test/test_custom.npy')
    print(test_set.shape, training_set.shape)
    # normalization of inputs
    train_mins = np.zeros(test_set.shape[1])
    train_mins *= 1000
    train_maxes = np.zeros(test_set.shape[1])
    test_mins = np.zeros(test_set.shape[1])
    test_mins *= 1000
    test_maxes = np.zeros(test_set.shape[1])
    for j in range(0,train_mins.shape):
        train_mins[j] = np.min(training_set[:, j])
        test_mins[j] = np.min(test_set[:, j])
    
    # tensor to dataloader
    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # autoencoder
    # set this flag to continue the training of model
    continue_training = True
    model_filename = 'data/pavencoder/' +  'model.pt' # model_filename = 'data/pavencoder/' +  'model.pt'
    checkpoint_filename = 'data/pavencoder/' + 'model_checkpoint.pt'
    if os.path.isfile(model_filename) and os.path.isfile(checkpoint_filename):
        if continue_training:
            print('Model and checkpoint files found. Loading.')
            [model, optimizer, checkpoint] = load_model_for_inference(model_filename, checkpoint_filename)
        else:
            print('If you want to resume training of model set the "continue_training=True."\n')
            print('If you want to start a new training model delete model.pt and model_checkpoint.pt from data/pavencoder folder.')
            print('Exiting program.')
            quit()
    else:
        print('Model not found. Creating...')
        input_size = training_set[0].shape[0] # size of each sample
        code_length = 128 # length of compressed version of each input
        model = Pavencoder(input_size=input_size, hidden=code_length)
        print('Done.')

    print('Moving model to', device)
    model.to(device)
    model.double()

    # train
    epochs = 100
    print('Training model for', epochs, 'epochs...')
    train(model, train_dataloader, test_dataloader, epochs=epochs, flatten=True)
    print('\nDone.')