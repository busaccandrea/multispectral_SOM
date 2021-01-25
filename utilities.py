import scipy.io as scio
import numpy as np
import os
from PIL import Image


def chem_el_tiff_to_array(imgfile):
    """
    read a chemical-element tiff img and returns a col-array
    """
    imgfile = Image.open(imgfile)
    img_array = np.array(imgfile)
    col_array = img_array.reshape([418*418])
    # transpose is uset to get a column vector
    return col_array.T


def get_matrix_from_chem_el_tiff(path_folder, save_to_file=True):
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


def save_clusters_images(clusters, base_filename='c'):
    """
    Given the matrix of clusters, save an image (png) for every cluster.
    The input is a matrix with shape (#pixels, #cluster).
    """
    results_folder = './results/'
    for k in range(0, clusters.shape[1]):# 0-> #clusters
        img_k = clusters[:,k]
        img_k = img_k.reshape(418, 418, order = 'F') # parameter 'F' is used to make the right position of image. (MATLAB reads arrays with a different order)
        image = Image.fromarray(img_k)
        image.save(results_folder + base_filename + str(k) + '.PNG')
