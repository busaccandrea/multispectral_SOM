from numpy.core.fromnumeric import ndim
from Pavencoder import train
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.utils import validation

""" script to build a dataset. """
data_filename = 'data/Edf20MS/data.npy'
data = np.load(data_filename)
d = data.reshape((130, 185, 2048))
testset_path = 'data/Edf20MS/test/*/'
trainset_path ='data/Edf20MS/train/*/'
testset_zones = glob(testset_path)
trainset_zones = glob(trainset_path)

def get_pixels_from_coords(points):
    row_start, row_end= points[0][0], points[1][0]
    col_start, col_end = points[0][1], points[1][1]
    temp = []
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            row = d.shape[1] * r + c
            temp.append(row)
    return np.array(temp) #list of spectra
    
def split_data(spectras):
    """ 
        Given a 2d-array this method splits 
     """
    [train_data, t] = ttsplit(spectras, test_size=0.3, train_size=0.7, shuffle=True)
    [validation_data, test_data] = ttsplit(t, train_size=0.66, test_size=0.33, shuffle=True)
    train_data = data[list(train_data)]
    test_data = data[list(test_data)]
    validation_data = data[list(validation_data)]

    return [train_data, validation_data, test_data]

def merge_dataset(datasets, _data):
    """ datasets =  [train_set, validation_set, test_set]
        _data = [test_data:ndarray, valid_data:ndarray, test_data:ndarray]
     """
    if datasets[0].ndim == 1:
        train_set = _data[0]
        validation_set = _data[1]
        test_set = _data[2]
    else:
        train_set = np.concatenate((datasets[0], _data[0]), axis=0)
        validation_set = np.concatenate((datasets[1], _data[1]), axis=0)
        test_set = np.concatenate((datasets[2], _data[2]), axis=0)
    return [train_set, validation_set, test_set]
    

def build_from_zero():
    # Importazione punti da file?
    zones = [[[0,0],[39,40]],
        [[0,47],[40,89]],
        [[0,99],[40,141]],
        [[0,147],[40,183]],
        [[47,0],[87,39]],
        [[47,45],[88,88]],
        [[44,96],[85,139]],
        [[46,146],[85,184]],
        [[95,0],[116,13]],
        [[96,17],[115,38]],
        [[97,44],[129,89]],
        [[97,97],[128,138]],
        [[97,143],[118,164]],
        [[96,169],[118,184]],
        [[120,0],[128,13]],
        [[121,18],[129,39]],
        [[121,18],[129,164]],
        [[121,168],[128,184]]]

    train_set = np.zeros(data[0].shape)
    validation_set = np.zeros(data[0].shape)
    test_set = np.zeros(data[0].shape)

    for zone_index, points in enumerate(zones):
        spectras = get_pixels_from_coords(points=points)
        [train_data, validation_data, test_data] = split_data(spectras)
        np.save(trainset_zones[zone_index] + '/train.npy', train_data)
        np.save(trainset_zones[zone_index] + '/validation.npy', validation_data)
        np.save(testset_zones[zone_index] + '/test.npy', test_data)

        datasets = [train_set, validation_set, test_set]
        _data = [train_data, validation_data, test_data]
        [train_set, validation_set, test_set] = merge_dataset(datasets=datasets, _data=_data)
        
    np.save('data/Edf20MS/train/train.npy', train_set)
    np.save('data/Edf20MS/train/validation.npy', validation_set)
    np.save('data/Edf20MS/test/test.npy', test_set)

def customize_dataset_zones(zones, edf_path='data/Edf20MS/'):
    train_set = np.zeros(data[0].shape)
    validation_set = np.zeros(data[0].shape)
    test_set = np.zeros(data[0].shape)

    # zones = 1,3,5,6,7,8,10
    for zone in zones:
        train_data = np.load(trainset_zones[zone-1] + '/train.npy')
        validation_data = np.load(trainset_zones[zone-1] + '/validation.npy')
        test_data = np.load(testset_zones[zone-1] + '/test.npy')

        datasets = [train_set, validation_set, test_set]
        _data = [train_data, validation_data, test_data]
        [train_set, validation_set, test_set] = merge_dataset(datasets=datasets, _data=_data)
    
    np.save(edf_path +'train/train_custom.npy', train_set)
    np.save(edf_path +'train/validation_custom.npy', validation_set)
    np.save(edf_path +'test/test_custom.npy', test_set)

# a = np.load('data/Edf20MS/train/train.npy')
# print(a.shape)


# zones = [1,2,4,5,8]
# customize_dataset_zones(zones=zones)


# build_from_zero()