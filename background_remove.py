# from scipy.signal import fftconvolve
import numpy as np
from matplotlib import pyplot as plt
from time import time

from numpy.core.shape_base import block
import find_peaks_

""" peak shape:
    G(x) = A exp (- ((x-a)^2) / (2sigma^2)), where A=intensity, a=position, sigma=std_dev

it is supposed that the background can be approximated by a linear function in a short interval. 
so if the interval is the spectra region, we can say that this portion is 
    y(x) = G(x) + B + Cx 
where b+cx is the background function.

this method is based on a double convolution of the spectrum with f'(x) = Der[exp(-(t - x)^2 / (2delta^2))].
convolving the spectrum with a gaussian function with right sigma we can obtain a spectrum already cleaned from the 
background. after the second convolution the resulting spectrum is a smooth line without background.

the mean problem with this approach is to choose the right kernel dimension (sigma). The best performance are obtained when 
kernel and peak have same dimension (delta = sigma). 

to find the right sigma we can compute multiple times convolutions with diverse sigma and compare all convoluted spectras. 
(get the maximum value for peak?) 

coded from following pubblication:
    https://iopscience.iop.org/article/10.1088/0022-3727/36/15/323
"""


def remove_bg(spectra, kernel):
    c1 = np.convolve(spectra, kernel, mode='same')#[:spectra.shape[0]]
    c2 = np.convolve(c1, kernel, mode='same')#[:spectra.shape[0]]
    c2[c2>=0] = 0
    c2 = -c2
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    # plt.plot(c1, 'g', c2, 'r')
    # plt.pause(0.001)
    # plt.show(block=True)
    return c2.astype(int)

print('Start. loading data')
# data
data = np.load('./data/Edf20MS/data.npy')
print('loaded. Computing sum spectra.')

# sample = data[1000:1002]
sample = data
sample[:, :50] = 0
sample[:, 1250:-1] = 0

# get all sigma for gaussian kernel
sigma_array = np.array([3])
gaussians = np.zeros((sigma_array.shape[0], data.shape[1])) # n.gaussians * n.channel
conv_results = np.zeros(data.shape)

kernels = gaussians.copy()
x = np.arange(0, gaussians.shape[1])

# if user wants to show certain row of data
index_to_show = []

print('background removing...')
for r, row in enumerate(data):
    if (len(index_to_show) > 0 and r in index_to_show) or len(index_to_show) == 0:
        for sigma_index, sigma in enumerate(sigma_array):

            kernels[sigma_index] = np.gradient(np.exp(-((x - 1024)/sigma)**2))
            conv_results[r] = remove_bg(row, kernels[sigma_index])

            if len(index_to_show) > 0:
                # plot to compare spectra pre and post bg-removing
                plt.figure(r)
                plt.ion()
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
                plt.plot(row, 'g', conv_results[r], 'r')
                plt.pause(0.001)
                plt.show(block=True)

np.save('./data/Edf20MS/data_nobg.npy', conv_results)

sum_conv_results = find_peaks_.sum_spectra(conv_results)

# getting sum spectra of dataset
data_sum_spectra = find_peaks_.sum_spectra(data)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.plot(data_sum_spectra, 'g', sum_conv_results, 'r')
plt.show(block=True)