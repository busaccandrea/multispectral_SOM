from pathlib import WindowsPath
from scipy import sparse
import scipy
import scipy.io as scio
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob


def chem_el_tiff_to_array(imgfile):
    """
    read a chemical-element tiff img and returns a col-array
    """
    imgfile = Image.open(imgfile)
    img_array = np.array(imgfile)
    col_array = img_array.reshape([418*418])
    # transpose is uset to get a column vector
    return col_array.T


def get_matrix_from_chem_el_tiff_(path_folder, save_to_file=True):
    """
    reads only the .tiff the path and builds a matrix [418*418, N], where N is the number of chem-elem-tiffs
    """
    listfiles = os.listdir(path_folder)
    M = np.zeros([418*418, len(listfiles)])
    for i, f in enumerate(listfiles):
        col_array = chem_el_tiff_to_array(path_folder + f)
        idx = int(f[0]) - 1
        M[:, idx] = col_array

    if save_to_file:
        np.save('M_from_chem_el_tiffs', M)
    
    return M

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


def save_clusters_images_old(clusters, base_filename='c'):
    """
    Given the matrix of clusters, save an image (png) for every cluster.
    The input is a matrix with shape (#pixels, #cluster).
    """
    for k in range(0, clusters.shape[1]):# 0-> #clusters
        img_k = clusters[:,k]
        img_k = img_k.reshape(418, 418, order = 'F') # parameter 'F' is used to make the right position of image. (MATLAB reads arrays with a different order)
        image = Image.fromarray(img_k)
        image.save('results/' + base_filename + str(k) + '.PNG')


def save_clusters_images(clusters_perc, basename='c'):
    for k in range(0, clusters_perc.shape[1]):# 0-> #clusters
        img_k = clusters_perc[:,k]
        img_k = img_k.astype(np.uint8)
        img_k = img_k.reshape(418, 418, order = 'F') # parameter 'F' is used to make the right position of image. (MATLAB reads arrays with a different order)
        image = Image.fromarray(img_k)
        # image.show()
        image.save('results/' + basename + str(k) + '.PNG')


# def get_clusters_spectras_old(clusters, write_in_file = False):
#     """ Compute the representative spectra of each cluster. """
#     data = np.load('./data/B.npy')
#     avg_spectras = np.zeros((clusters.shape[1], 2048))
    
#     # for each cluster
#     for i, row in enumerate(clusters):
#         j = np.argmax(row) # accendo il pixel i nel gruppo j se i ha valore massimo nella colonna j.
        
#         # aggiungo lo spettro del pixel i allo spettro cumulativo del gruppo j
#         avg_spectras[j] += data[i]

#     # divido ogni spettro del gruppo j per il corrispondente numero di pixel accesi
#     for j in range(0, avg_spectras.shape[0]):
#         avg_spectras[j] = avg_spectras[j] / np.count_nonzero(clusters[:, j])
#         np.save('./results/B/avg_spectra_' + str(j) + '.npy', avg_spectras[j])
#         plt.clf()
#         plt.plot(avg_spectras[j])
#         plt.show()
#         if write_in_file:
#             plt.savefig('./results/B/avg_spectra_' + str(j) + '.png', dpi=300)
    
#     return avg_spectras

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

        np.save('results/' + filename + 'avg_perc_spectra_' + str(j) + '.npy', spectras[j])
        plt.clf()
        plt.plot(spectras[j])
        if write_in_file:
            plt.savefig('./results/' + filename + '/' + 'avg_perc_spectra_' + str(j) + '.png', dpi=300)
        # plt.show()
    
    return spectras
