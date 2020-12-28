import scipy.io as scio
import numpy as np
from scipy import sparse
import scipy.signal as sci_sig
import matplotlib.pyplot as plt

def max_spectra(spectra, write_in_file = False):
    """
    Compute the max spectra of input spectra.
    If you want to save in a file
    """
    max_spectra = np.zeros(2048)
    for col in range(0, 2048):
        max_spectra[col] = np.max(B_[:,col])
    
    if write_in_file:
        np.save('max_spectra.npy', max_spectra)
    
    return max_spectra

def sum_spectra(spectra, write_in_file=False):
    """
    Compute the sum spectra of input.
    """
    sum_spectra = np.zeros(2048)
    for col in range(0, 2048):
        sum_spectra[col] = np.sum(B_[:,col])

    if write_in_file:
        np.save('sum_spectra.npy', sum_spectra)
    
    return sum_spectra

def find_peaks_(spectra, threshold=[3, None], width=3):
    """
    Find peaks in spectra given a threshold and a width.
    """
    # peaks is a ndarray of indices of peaks in spectra
    peaks = sci_sig.find_peaks(spectra, threshold=[3, None], width=3)

    # obtain a ndarray of peaks.
    output = np.zeros(len(spectra))
    for i in peaks[0]:
        output[i] = spectra[i]
    plt.plot(sum_spectra, 'b', output, 'r')
    plt.show()

# # per caricare un file .mat usare la libreria di scipy.io
mat_file = scio.loadmat('./B.mat')
B_ = mat_file["B"]
B = np.zeros(B_.shape)



# determino lo spettro somma da file




# plt.plot(max_spectra)
# plt.show()

spectra = sum_spectra
# avg = np.average(sum_spectra)




# # filtraggio picchi
# for idx, row in enumerate(B_):
#     peaks = find_peaks(row, distance=6, threshold=2.0)[0]
#     for peak_idx in peaks:
#         for i in range(peak_idx - 3, peak_idx + 3):
#             B[idx][i] = row[i]