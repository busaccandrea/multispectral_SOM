import numpy as np
import glob
from pymca import EdfFile
from scipy import sparse
from utilities import check_existing_folder
from time import time
 
def get_data_from_edf_file(edf_file):
    edf = EdfFile(edf_file)
    return edf.GetData(0)

def pack_edf_files(source_path='./', output_path='./', save_sparse=False, step_length=0):
    filelist = glob.glob(source_path + '*.edf')
    if len(filelist) == 0:
        print('Wrong path: edf files not found.')
        quit()
    check_existing_folder(output_path)
    datashape = get_data_from_edf_file(filelist[0]).shape

    if step_length == 0:
        step_length = len(filelist * datashape[0])

    for i, edf_file in enumerate(filelist):
        start = time()
        d = get_data_from_edf_file(edf_file=edf_file)
        if i % step_length == 0:
            data = d 
        else:
            data = np.concatenate((data, d), axis=0)
        if i % step_length == step_length - 1:
            print('provo a salvare la matrice di dimensioni:', data.shape)
            if save_sparse:
                sparse.save_npz(output_path + 'sparse_data_' + str(i+1) + '.npz', sparse.csr_matrix(data))
            else:
                np.save(output_path + 'data_' + str(i+1) + '.npy', data)
            print('salvata.')
            # data = np.zeros((datashape[0]*len(filelist), datashape[1]))
        print('file number:', i, "{:.2f}".format(i/len(filelist)*100), '%',' - elapsed time:', "{:.2f}".format(time() - start), end='\r')
    return data

pack_edf_files(source_path='./data/Braque/Edf/', output_path='./data/Braque/', save_sparse=True, step_length=50)