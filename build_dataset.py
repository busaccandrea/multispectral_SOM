from __future__ import with_statement
from numpy.core.fromnumeric import argmin, ndim
from Pavencoder import train
import numpy as np
import csv
from glob import glob
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.utils import validation
from utilities import get_area_coordinates
from PIL import Image

""" script to build a dataset. """
# data_filename = 'data/Edf20MS/data.npy'
# data = np.load(data_filename)
# d = data.reshape((130, 185, 2048))
# testset_path = 'data/Edf20MS/test/*/'
# trainset_path ='data/Edf20MS/train/*/'
# testset_zones = glob(testset_path)
# trainset_zones = glob(trainset_path)

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
""" 
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
coefficients = [1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 8,
 4,
 2,
 2,
 4,
 8,
 16,
 8,
 2,
 16]

data =  np.load('./data/Edf20MS/data.npy')
dataset = data[0]
dataset = np.expand_dims(dataset, axis=0)
next2label = 0
for idx, zone in enumerate(zones):
    point0, point3 = zone
    r = get_area_coordinates(data, point0=point0, point3=point3)
    x,y,z = r.shape
    r = r.reshape((x*y, z))
    zon = r.copy()
    print('zone',idx+1, x*y*coefficients[idx])
    for _ in range(0, coefficients[idx]):
        dataset =  np.concatenate((dataset, r), axis=0)
        zon = np.concatenate((zon, r), axis=0)
    np.save('./data/Edf20MS/zones/'+str(idx)+'.npy', zon)
np.save('./data/Edf20MS/zones/data2label.npy', dataset)

# dataset for class 9 (Co) """

""" # dataset for element VS all approach
if __name__ == '__main__':
    el_distribution = np.array([
        [1,	0,	1,  1,  0,  1,  0,	0,	0,	0],
        [1,	1,	0,	0,	0,	1,	1,	0,	0,	0],
        [1,	1,	1,	1,	0,	1,	0,	0,	0,	0],
        [1,	1,	0,	1,	0,	1,	0,	0,	0,	0],
        [1,	1,	1,	1,	1,	1,	0,	0,	1,	0],
        [1,	1,	1,	1,	1,	0,	0,	0,	1,	0],
        [1,	1,	0,	0,	0,	1,	1,	0,	1,	0],
        [1,	1,	1,	1,	0,	1,	0,	0,	1,	0],
        [1,	1,	1,	1,	0,	1,	0,	1,	1,	0],
        [1,	1,	0,	0,	0,	1,	0,	0,	1,	0],
        [1,	1,	0,	0,	0,	0,	0,	0,	1,	0],
        [1,	1,	0,	1,	0,	1,	0,	1,	1,	1],
        [1,	1,	1,	1,	0,	1,	0,	0,	0,	0],
        [1,	1,	1,	0,	0,	1,	0,	0,	0,	0],
        [1,	1,	0,	0,	0,	0,	0,	0,	1,	0],
        [1,	0,	0,	0,	0,	0,	0,	0,	1,	0],
        [1,	1,	1,	1,	0,	1,	0,	0,	1,	0],
        [1,	1,	1,	1,	1,	1,	0,	0,	1,	1]])
    # datasets = [] #lista di matrici
    for element in range(1,10):# build all datasets except Ca (element 0)
        # aggiungo al dataset solo le zone in cui c'è la presenza di element.
        # queste zone sono le righe in cui alla colonna element hanno 1.
        print('\n\nelement:', element)
        
        with open('./data/Edf20MS/classes/'+str(element)+'.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ')
            
            zones_1 = []; zones_0 = []
            for zone, v in enumerate(el_distribution[:, element]): # determino in quale zona è contenuto element
                if v==1:
                    zones_1.append(zone)
                else:
                    zones_0.append(zone)

            # datasets.append(1)
            first = True
            for zone in zones_1:
                tmp = np.load('./data/Edf20MS/zones/'+str(zone)+'.npy')
                if first:
                    positive_class = tmp
                    first = False
                else:
                    positive_class = np.concatenate((positive_class, tmp), axis=0)

            first = True
            for zone in zones_0:
                tmp = np.load('./data/Edf20MS/zones/'+str(zone)+'.npy')
                if first:
                    negative_class = tmp
                    first = False
                else:
                    negative_class = np.concatenate((negative_class, tmp), axis=0)
            
            # del dataset di classe positiva devo prendere tutti i campioni.
            # nel dataset finale ci devono essere 2*positive_class.shape campioni,
            # dove il 50% sono positivi e il restante 50% sono negativi.
            # se negative.shape < positive_class.shape allora devo duplicare elementi di negative_class.
            n_positives = positive_class.shape[0]
            print('shape of positive_class:', n_positives)
            print('shape of negative_class:', negative_class.shape[0])
            while negative_class.shape[0] < n_positives:
                print('\tother', n_positives-negative_class.shape[0], 'needed.')
                negative_class = np.concatenate((negative_class, negative_class), axis=0)
                print('\tnegative_class updated:', negative_class.shape[0])

            print('writing all samples from positive_class on csv.')
            # write csv for all ones.
            for row1, _ in enumerate(positive_class):
                csv_writer.writerow([row1, 1])

            print('writing', n_positives, 'elements from negative_class on csv.')
            # write csv for all zeros.
            np.random.shuffle(negative_class)
            negative_class = negative_class[:n_positives]
            for row0, _ in enumerate(negative_class):
                csv_writer.writerow([row0+row1, 0])

            # np.save('./data/Edf20MS/classes/'+str(element)+'_1.npy', positive_class)
            # np.save('./data/Edf20MS/classes/'+str(element)+'_0.npy', negative_class)
            dataset = np.concatenate((positive_class, negative_class), axis=0)
            print('final dataset shape:', dataset.shape, '\npositive_class shape:', positive_class.shape,'\nnegative_class shape:', negative_class.shape)
            np.save('./data/Edf20MS/classes/'+str(element)+'.npy',dataset) """

# build the dataset for ChemElementRegressor
# i need to cerrespond a number for each row of the dataset.
# the number is taken by the maps from pymca
if __name__ == '__main__':

    data_filename = 'data/DOggionoGiulia/EDF/data.npy'
    data = np.load(data_filename)
    print(data.shape[0], data.shape[1])
    elements = ['Ca-K','Cu-K','Fe-K','Hg-L','K-K','Mn-K','Pb-L','Sn-L','Sr-K','Ti-K']
    for i, element in enumerate(elements):
        # load the element map and convert it into a 1d-array
        f = './data/DOggionoGiulia/float32/'+element+'.tif'
        element_tiff = Image.open(f)
        element_map = np.array(element_tiff)
        element_map = element_map.reshape((element_map.shape[0] * element_map.shape[1]))
        print(element, 'is', f)
        
        # create the csv file.
        with open('./data/DOggionoGiulia/'+str(element)+'.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ')

            for i, row in enumerate(data):
                # c = 1 if element_map[i] > 200 else 0
                # generic row: i-th_row n_counts
                csv_writer.writerow([i, element_map[i]])