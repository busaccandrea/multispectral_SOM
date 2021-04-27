import scipy.io as scio
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt


def chem_el_tiff_to_array(imgfile):
    """
    read a chemical-element tiff img and returns a col-array
    """
    imgfile = Image.open(imgfile)
    img_array = np.array(imgfile)
    col_array = img_array.reshape([418*418])
    # transpose is uset to get a column vector
    return col_array.T


def get_matrix_from_chem_el_tiffs(file_list, save_to_file=True):
    """
    reads only the .tiff the path and builds a matrix [418*418, N], where N is the number of chem-elem-tiffs
    """
    M = np.zeros([418*418, len(file_list)])
    for i, f in enumerate(file_list):
        col_array = chem_el_tiff_to_array(f)
        M[:, i] = col_array

    if save_to_file:
        np.save('M_from_chem_el_tiffs', M)
    
    return M

    
def get_matrix_from_matfile(matfile, key_dict=''):
    """
    This method is made to get a matrix in a .mat file.
    If key_dict is given returns the ndarray inside it.
    Else returns a dict file.

    key_dict: the matfile is a dictionary. with key_dict you can load the related matrix from the matfile.
    """
    if os.path.isfile(matfile):
        mat_file = scio.loadmat(matfile)
    else:
        print('this matfile does not exists.')
        return None
    
    # must return a ndarray
    if not key_dict == '':
        M = mat_file[key_dict]
        return M
    else: # must return a dict
        return mat_file


def get_coord_from_pixel(pixel:int):
    """
    Cast the pixel into two coords (i,j).
    Use this method to transform the row from the matrix of spectra (e.g. 125000 x 2048) to (i,j) coordinates, with i,j = 0,1,...,417.
    """
    row = int(pixel/418)
    col = pixel % 418
    return row, col


def save_clusters_images(clusters_perc, basename='c'):
    for k in range(0, clusters_perc.shape[1]):# 0-> #clusters
        img_k = clusters_perc[:,k]
        img_k = img_k.astype(np.uint8)
        img_k = img_k.reshape(418, 418, order = 'F') # parameter 'F' is used to make the right position of image. (MATLAB reads arrays with a different order)
        image = Image.fromarray(img_k)
        # image.show()
        check_existing_folder('results/' + basename)
        image.save('results/' + basename + str(k) + '.PNG')


def get_clusters_spectras(clusters, allow_average=False, write_in_file = False, filename=''):
    """ Compute the representative spectra of each cluster. """
    data = np.load('./data/B.npy')
    spectras = np.zeros((clusters.shape[1], 2048)) # [8, 2048]
    
    for pixel, perc_array in enumerate(clusters):
        for cluster_index, _ in enumerate(spectras):
            spectras[cluster_index] += data[pixel] * perc_array[cluster_index]

    # divido ogni spettro del gruppo j per il corrispondente numero di pixel accesi
    for j in range(0, spectras.shape[0]):
        if allow_average:
            spectras[j] = spectras[j] / np.sum(clusters[:, j])
        
        # plt.plot(spectras[j])

        check_existing_folder('results/' + filename)
        np.save('results/' + filename + 'avg_perc_spectra_' + str(j) + '.npy', spectras[j])
        plt.clf()
        plt.plot(spectras[j])
        if write_in_file:
            plt.savefig('./results/' + filename + '/' + 'avg_perc_spectra_' + str(j) + '.png', dpi=300)
        # plt.show()
    
    return spectras

def check_existing_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Folder not found, created.')

def get_area_coordinates(data, point0, point3):
    """ given two data points (point0 and point3), returns the point set of area between those points.
        it's used to build a balanced dataset and to replicate a little set of spectra.

        point0 ._____________.  point1
               |             |
               |             |
        point2 !_____________! point3
                                                    """

    (p0x, p0y) = point0
    (p3x, p3y) = point3
    
    # point1
    p1x = min(p0x,p3x)
    p1y = max(p0y,p3y)
    
    # point2
    p2x = max(p0x,p3x)
    p2y = min(p0y,p3y)

    d = data.reshape((130, 185, data.shape[1])) #the shape of square data

    return d[p0x: p2x, p0y:p3y]

"""             point0   |  point3
 1  * zona 1:
 1  * zona 2:   
 1  * zona 3:
 1  * zona 4:
 1  * zona 5:
 1  * zona 6:
 1  * zona 7:
 1  * zona 8:
 8  * zona 9:   94,0        117,14
 4  * zona 10:  95,17       117,39
 2  * zona 11:  95,42       129,90
 2  * zona 12:  95,95       129,140
 4  * zona 13:  96,142      118,165
 8  * zona 14:  96,169      118,184
16  * zona 15:  119,0       129,14
 8  * zona 16:  119,17      39,129
 8  * zona 17:  119,142     165,129
16  * zona 18:  120,167     129,184
                                        """
""" data = np.load('./data/Edf20MS/data_nobg.npy')
new_data = data.copy()

# zona 9
zone9 = get_area_coordinates(data, (94,0), (117,14))
x, y, z = zone9.shape
zone9 = zone9.reshape((x*y, z))
for _ in range(8):
    new_data = np.concatenate((new_data, zone9), axis=0)

# zona10
zone10 = get_area_coordinates(data, (95,17), (117,39))
x, y, z = zone10.shape
zone10 = zone10.reshape((x*y, z))
for _ in range(4):
    new_data = np.concatenate((new_data, zone10), axis=0)

# zone11
zone11 = get_area_coordinates(data, (95,42), (129,90))
x, y, z = zone11.shape
zone11 = zone11.reshape((x*y, z))
for _ in range(2):
    new_data = np.concatenate((new_data, zone11), axis=0)

# zone12
zone12 = get_area_coordinates(data, (95,95 ), (129,90))
x, y, z = zone12.shape
zone12 = zone12.reshape((x*y, z))
for _ in range(2):
    new_data = np.concatenate((new_data, zone12), axis=0)

# zone13
zone13 = get_area_coordinates(data, (96,142), (118,165))
x, y, z = zone13.shape
zone13 = zone13.reshape((x*y, z))
for _ in range(4):
    new_data = np.concatenate((new_data, zone13), axis=0)

# zone14
zone14 = get_area_coordinates(data, (96,142), (118,184))
x, y, z = zone14.shape
zone14 = zone14.reshape((x*y, z))
for _ in range(8):
    new_data = np.concatenate((new_data, zone14), axis=0)

# zone15
zone15 = get_area_coordinates(data, (119,0), (129,14))
x, y, z = zone15.shape
zone15 = zone15.reshape((x*y, z))
for _ in range(16):
    new_data = np.concatenate((new_data, zone15), axis=0)

# zone16
zone16 = get_area_coordinates(data, (119,17), (39,129))
x, y, z = zone16.shape
zone16 = zone16.reshape((x*y, z))
for _ in range(8):
    new_data = np.concatenate((new_data, zone16), axis=0)

# zone17
zone17 = get_area_coordinates(data, (119,142), (165,129))
x, y, z = zone17.shape
zone17 = zone17.reshape((x*y, z))
for _ in range(8):
    new_data = np.concatenate((new_data, zone17), axis=0)

# zone18
zone18 = get_area_coordinates(data, (120,167), (129,184))
x, y, z = zone18.shape
zone18 = zone18.reshape((x*y, z))
for _ in range(16):
    new_data = np.concatenate((new_data, zone18), axis=0)

print(data.shape, new_data.shape)
np.save('./data/Edf20MS/data_nobg_enhanced.npy', new_data) """