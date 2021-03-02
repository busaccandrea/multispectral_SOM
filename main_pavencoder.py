import matplotlib.pyplot as plt
from scipy import sparse
import torch
import os
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from Pavencoder import load_model_for_inference
import time

""" 
    This script needs a trained model!
"""
if __name__=='__main__':

    # Load model
    model_filename = 'data/pavencoder/' +  'model.pt'
    checkpoint_filename = 'data/pavencoder/' + 'model_checkpoint.pt'
    if os.path.isfile(model_filename) and os.path.isfile(checkpoint_filename):
        print('Model and checkpoint files found. Loading.')
        [model, optimizer, checkpoint] = load_model_for_inference(model_filename, checkpoint_filename)

    # model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Model loaded and moved to :', device)

    # define batch size for AE
    batch_size = 64

    # load data file
    data_np = np.load('data/data.npy')
    # data to tensor
    data = torch.tensor(data_np).float()
    sum_dec = 0 * data[0]

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    model.eval()

    for row in data:
        code = model.encoder(row)
        decoded = model.decoder(code)
        sum_dec += decoded

    sum_data = np.sum(data_np, axis=0)

    plt.plot(sum_data, 'g-', sum_dec.detach().numpy(), 'r^')
    plt.show()