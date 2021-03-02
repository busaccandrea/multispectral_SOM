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


if __name__ == '__main__':

    # define which device is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define batch size for AE
    batch_size = 64

    # data file name to use for training
    data_filename = 'data/data.npy'
    
    # split data in two 2-dimensional tensors: 70% training set, 30% test set.
    [training_set, test_set] = split_data_from_numpy(data_filename)
    
    # tensor to dataloader
    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # autoencoder
    # set this flag to continue the training of model
    continue_training = True
    model_filename = 'data/pavencoder/' +  'model.pt'
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
        print('No model found. Creating...')
        input_size = training_set[0].shape[0] # size of each sample
        code_length = 8 # length of compressed version of each input
        model = Pavencoder(input_size=input_size, hidden=code_length)
        print('Done.')

    print('Moving model to', device)
    model.to(device)

    # train
    epochs = 10
    print('Training model for', epochs, 'epochs...')
    train(model, train_dataloader, test_dataloader, epochs=epochs, flatten=True)
    print('\nDone.')