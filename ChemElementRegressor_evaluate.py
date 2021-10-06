import torch
import numpy as np
from PIL import Image
from time import time
from sklearn.preprocessing import normalize
from torch.functional import norm
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from ChemElementRegressorDataset_for_evaluate import ChemElementRegressorDataset_for_evaluate
from glob import glob
from os import path

from utilities import check_existing_folder

""" 
    This script needs a trained model!
"""
if __name__=='__main__':
    # define device to use
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device available:', device)
    running_times = []

    data_loading_time = time()
    print('loading data...')
    data_np = np.load('data.npy')
    print('loaded in', time() - data_loading_time)


    print('creating dataset and loader...')
    dataset_time = time()
    data_set = ChemElementRegressorDataset_for_evaluate(data=data_np)
    eval_loader = DataLoader(dataset=data_set, batch_size=1024, shuffle=False)
    print('created in', time() - dataset_time)


    # elements = ['Ca','Cu','Fe','Hg','K','Mn','Pb','Sn','Sr','Ti']
    exclude = ['Ca','Cu','Fe','Hg', 'K', 'Mn', 'Pb','Sn']
    elements = glob('data/ChemElementRegressor/*.p')
    for element in elements:
        chemel = path.basename(element).split('.')[0]
        print('evaluating', chemel)
        if chemel in exclude:
            continue
        element_timestart = time()
        model = torch.load(element)
        model.to(device)
        model.eval()
        print('Model loaded and moved to :', device)
        
        batch_size = 1024

        first_time = True

        for i, batch_ in enumerate(eval_loader):
            print('batch:', i,'/', int(data_set.__len__()/batch_size), end='\r')
            batch_ = batch_.to(device)
            tmp_output = model(batch_)
            tmp_output = torch.squeeze(tmp_output).to('cpu').detach()
            tmp_output[tmp_output<0] = 0
            if first_time:
                outputs = tmp_output
                first_time = False
            else:
                outputs = torch.cat((outputs, tmp_output), dim=0)
            
        running_times.append(time() - element_timestart)
        print('\n',element,'evaluated.')

        outputs = outputs.numpy()
        outputs = outputs.reshape((1500,1960)) #braque shape

        check_existing_folder('./data/Braque/eval/')
        np.save('./data/Braque/eval/' + chemel + '_evaluated.npy', outputs)

        ground_truth = np.array(Image.open('./data/Braque/labels/'+ chemel + '.tiff'))
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
        print('Image saved in path:', './results/' + str(element) + '.tiff')
        
    running_times = np.array(running_times)
    for element, t in enumerate(running_times):
        print('element', element, 'evaluated in ', t)
    print('\nall elements evaluated in ', np.sum(running_times))