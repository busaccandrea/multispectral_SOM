
import numpy as np
import glob
from pymca import EdfFile
from scipy import sparse
 
def get_data_from_edf_file(edf_file):
    edf = EdfFile(edf_file)
    return edf.GetData(0)

def pack_edf_files(path_to_files='./'):
    filelist = glob.glob(path_to_files + '*.edf')
    if len(filelist) == 0:
        print('Wrong path: edf files not found.')
        return None
    datashape = get_data_from_edf_file(filelist[0]).shape
    data = np.zeros((datashape[0]*len(filelist), datashape[1]))
    print(data.shape)
    for i, edf_file in enumerate(filelist):
        d = get_data_from_edf_file(edf_file=edf_file) # d.shape = 418, 2048
        data[i * d.shape[0] : ((i+1) * d.shape[0])] = d

    sparse_data = sparse.csr_matrix(data)
    sparse.save_npz('sparse_data.npz', sparse_data)
    np.save('data/1data.npy', data)
    return data
