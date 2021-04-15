import scipy.io as scio
import numpy as np
from scipy import sparse
import scipy.signal as sci_sig
import matplotlib.pyplot as plt

def max_spectra(spectra_matrix, write_in_file = False):
    """
    Compute the max spectra of input.
    If you want to save in a file
    Input is the matrix of spectra.
    """
    max_spectra = np.zeros(2048)
    for col in range(0, 2048):
        max_spectra[col] = np.max(spectra_matrix[:,col])
    
    if write_in_file:
        np.save('max_spectra.npy', max_spectra)
    
    return max_spectra

def sum_spectra(spectra_matrix, write_in_file=False):
    """
    Compute the sum spectra of input.
    Input is the matrix of spectra.
    """
    # sum_spectra = np.zeros(spectra_matrix.shape[1])
    # for col in range(0, sum_spectra.shape[1]):
    #     sum_spectra[col] = np.sum(spectra_matrix[:,col])

    sum_spectra = np.sum(spectra_matrix, axis=0)
    
    if write_in_file:
        np.save('sum_spectra.npy', sum_spectra)
    
    return sum_spectra

def find_and_plot_peaks(spectra, threshold=[3, None], width=3):
    """
    Find peaks in spectra given a threshold and a width.
    """
    # peaks is a ndarray of indices of peaks in spectra
    peaks = sci_sig.find_peaks(spectra, threshold= threshold, width=width, distance=None)
    print('# of peaks found:', len(peaks[0]))
    # obtain a ndarray of peaks.
    output = np.zeros(len(spectra))
    count = 0
    for i in peaks[0]:
        if i > 212 and i < 1300:
            output[i] = spectra[i]
            count += 1
    print('# of peaks plotted = ', count)
    plt.plot(spectra, 'bo', output, 'r', markersize=2)
    plt.show()

    return peaks, output

# # per caricare un file .mat usare la libreria di scipy.io
# mat_file = scio.loadmat('./B.mat')
# B_ = mat_file["B"]



# # filtraggio picchi
# for idx, row in enumerate(B_):
#     peaks = find_peaks(row, distance=6, threshold=2.0)[0]
#     for peak_idx in peaks:
#         for i in range(peak_idx - 3, peak_idx + 3):
#             B[idx][i] = row[i]