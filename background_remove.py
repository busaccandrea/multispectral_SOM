# from scipy.signal import fftconvolve
import numpy as np
from matplotlib import pyplot as plt
from time import time
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
    der = np.gradient(c2)
    der = np.gradient(der)
    """ zero_crossings = zero_crossing(c2)
    t = np.zeros(c1.shape)
    t-=20
    t[zero_crossings]=0 """
    c2[c2>=0] = 0
    c2 = -c2
    c2[c2<2] = 0
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.plot(c1, 'g', c2, 'r', der, 'y')
    plt.pause(0.001)
    plt.show(block=True)
    return [c1.astype(int), c2.astype(int)]

""" def remove_bg__(spectra, kernels):
    c = np.zeros((kernels.shape))
    for i, kernel in enumerate(kernels):
        c[i] = np.convolve(spectra, kernel, mode='same')#[:spectra.shape[0]]
        c[i] = np.convolve(c[i], kernel, mode='same')#[:spectra.shape[0]]
    
    c[c>=0] = 0
    c = -c
    return c.astype(int) """

# if __name__ == '__main__':
def test():
    print('Start. loading data')
    # data
    data = np.load('./data/Edf20MS/data.npy')
    print('loaded. Computing sum spectra.')

    # sample = data[1000:1002]
    sample = data
    sample[:, :50] = 0
    sample[:, 1250:-1] = 0

    # get all sigma for gaussian kernel
    sigma_array = np.array([3, 5, 7])
    gaussians = np.zeros((sigma_array.shape[0], data.shape[1])) # n.gaussians * n.channel
    conv_results = np.zeros((data.shape[0], data.shape[1], sigma_array.shape[0]))


    kernels = gaussians.copy()
    x = np.arange(0, gaussians.shape[1])

    # if user wants to show certain row of data
    index_to_show = []

    print('background removing...')
    for r, row in enumerate(data):
        if (len(index_to_show) > 0 and r in index_to_show) or len(index_to_show) == 0:
            for sigma_index, sigma in enumerate(sigma_array):
                kernels[sigma_index] = np.gradient(np.exp(-((x - 1023)/sigma)**2))
                conv_results[r,:,sigma_index] = remove_bg(row, kernels[sigma_index])

                if len(index_to_show) > 0:
                    # plot to compare spectra pre and post bg-removing
                    plt.figure(r)
                    plt.ion()
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                    plt.plot(row, 'g', conv_results[r], 'r')
                    plt.pause(0.001)
                    plt.show(block=True)

    # np.save('./data/Edf20MS/data_nobg_11.npy', conv_results)

    # sum_conv_results = find_peaks_.sum_spectra(conv_results)

    # getting sum spectra of dataset
    data_sum_spectra = find_peaks_.sum_spectra(data)

    colors = ['r','g','b','k']
    for sigma_index,_ in enumerate(sigma_array):
        sum_conv_results = find_peaks_.sum_spectra(conv_results[:,:,sigma_index])
        plt.plot(sum_conv_results, colors[sigma_index])
    plt.plot(data_sum_spectra, colors[3])
    plt.show()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.plot(data_sum_spectra, 'g', sum_conv_results, 'r')
    plt.show(block=True)

def gaussian(sigma, kernel_size=2048):
    x = np.arange(0, kernel_size)
    return np.gradient(np.exp(-((x - 1023)/sigma)**2))

import colorsys
def generate_colorlist(n_of_colors):
    HSV_tuples = [(x*1.0/n_of_colors, 1, 0.7) for x in range(n_of_colors)]
    RGB_tuples =list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    return RGB_tuples

def zero_crossing(spectra):
    return  np.where(np.diff(np.sign(spectra)))[0]

if __name__=='__main__':
    data = np.load('./data/Edf20MS/data.npy')[24000]
    # generate all different sigmas
    sigmas = np.array(range(10,11))
    start = time()
    # generate all kernels
    kernels = list(map(gaussian, sigmas))
    now = time()
    print('gaussians generated in:', now-start)
    conv_results = np.array(kernels.copy())
    _2nd_conv_results = np.array(kernels.copy())

    RGB_tuples = generate_colorlist(n_of_colors=sigmas.shape[0])

    legend_names = []

    plt.plot(data, 'k', linewidth=4)
    legend_names.append('Spectrum')
    start = time()
    for i, kernel in enumerate(kernels):
        conv_results[i], _2nd_conv_results[i] = remove_bg(data, kernel)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        legend_names.append('sigma=' + str(sigmas[i]))
        plt.plot(_2nd_conv_results[i], color=RGB_tuples[i])
    plt.legend(legend_names)
    now = time()
    print('all convolutions done in:', now-start)

    # mean_spectra = np.median(_2nd_conv_results, axis=0)
    # legend_names.append('mean')
    # plt.plot(mean_spectra, color=(0.5, 0.5, 0.5), linewidth=4)
    # plt.legend(legend_names)


    plt.show()