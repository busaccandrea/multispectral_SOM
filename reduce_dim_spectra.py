from scipy.signal import find_peaks
import find_peaks_
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

results_base_folder = './results/'
def peak_based_red(data, write_to_file=False):
    """
    Reduce dimensionality of data based on peaks found.
    """
    sum_spectra = find_peaks_.sum_spectra(data)
    peaks = find_peaks(sum_spectra, threshold=[250, None], width=5)[0]
    resized_data = np.zeros([data.shape[0], len(peaks)])
    for row in range(0, data.shape[0]):
        for col, peak_coord in enumerate(peaks):#      from \/          to \/
            resized_data[row][col] = np.sum(data[row][peak_coord - 5 : peak_coord + 5])
    
    if write_to_file:
        np.save('data_peaks_' + str(len(peaks)) + '.npy', resized_data)

    return resized_data


def pca_based(data, n_components_=15, write_to_file=False):
    """
    reduce dimensionality of data using PCA.
    """
    pca = PCA(n_components=n_components_)
    data_ = pca.fit_transform(data)

    pca_explained_variance =  pca.explained_variance_
    pca_components =  pca.components_
    pca_ratio = pca.explained_variance_ratio_

    if write_to_file:
        np.save(results_base_folder + 'pca/data_pca_' + str(n_components_) + '_comp.npy', data)
    

    return [data_, pca_explained_variance, pca_components, pca_ratio]
    

def nnmf_based(data, smoothing=0, n_components_=15, write_to_file=False, max_iterations=200):
    """
    Reduce dimensionality of data using NNMF.
    """
    data_ = data.copy()
    if smoothing > 0:
        print('Smoothing...')
        for row in range(0, data.shape[0]):
        # utilizzo la convoluzione per determinare in modo efficiente la media mobile sulle righe di B
            data_[row] = np.convolve(data[row], np.ones(smoothing), 'same') / smoothing
    print('Calculating...')
    nmf = NMF(n_components=n_components_, init='random', max_iter=max_iterations)
    W = nmf.fit_transform(data_)
    H = nmf.components_
    if write_to_file:
        np.save(results_base_folder + 'nmf/data_nmf_W_' + str(n_components_) + '_comp_smth_' + str(smoothing) + '.npy', W)
        np.save(results_base_folder + 'nmf/data_nmf_H_' + str(n_components_) + '_comp_smth_' + str(smoothing) + '.npy', H)
    
    return [W,H]
