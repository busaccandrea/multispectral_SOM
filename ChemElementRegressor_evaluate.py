from math import sqrt
import torch
import numpy as np
from PIL import Image
from time import time
from sklearn.preprocessing import normalize
from torch.functional import norm
from matplotlib import pyplot as plt

""" 
    This script needs a trained model!
"""
if __name__=='__main__':
    # define device to use
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_times = []
    
    elements = [
        1,
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
        element_timestart = time()
        model = torch.load('data/ChemElementRegressor/DOggionoGiuliaCa-Kconv.p')
        model.to(device)
        model.eval()
        print('Model loaded and moved to :', device)
        
        # load data file
        data_np = np.load('data/DOggionoGiulia/EDF/data.npy')

        # data to tensor
        data = torch.tensor(data_np)
        
        # outputs = torch.zeros(data.shape[0], requires_grad=False)
        outputs = np.zeros(data.shape[0])
        for i, row in enumerate(data):
            print('evaluating data...', int(i/data.shape[0]*100), '%', end='\r')
            # feed the model with current row. i-th result is in the i-th row and m-th column.
            row = torch.unsqueeze(row, dim=0)
            row = torch.unsqueeze(row, dim=1)
            row = row.to(device)
            model(row).to('cpu').detach()
            tmp_output = model(row).to('cpu').detach()
            # print(tmp_output)
            tmp_output[tmp_output<0] = 0
            outputs[i] = tmp_output
        running_times.append(time() - element_timestart)
        print('\nelement', element, 'evaluated.')

        outputs = outputs.reshape((418,418))

        # img = Image.fromarray(outputs)
        ground_truth = np.array(Image.open('./data/Edf20MS/classes/' + str(element) +'.tif'))
        f1 = plt.figure(1)
        plt.imshow(ground_truth)
        f1.show()
        # plt.imshow(img)
        f2 = plt.figure(2)
        plt.imshow(outputs)
        # plt.imshow(np.array('./data/Edf20MS/classes/'+str(element) +'.tif'))
        f2.show()
        # img.save('./results/' + str(element) + '.tif')
        input()
        print('Image saved in path:', './results/' + str(element) + '.tif')
        
    running_times = np.array(running_times)
    for element, t in enumerate(running_times):
        print('element', element, 'evaluated in ', t)
    print('\nall elements evaluated in ', np.sum(running_times))