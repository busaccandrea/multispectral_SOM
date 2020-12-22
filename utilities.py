import scipy.io as scio
import numpy as np
import os


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