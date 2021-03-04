import scipy.io as scio
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import edf_read
import time

# campione = np.load('data/Edf20MS/Edf/data.npy')
a = np.load('data/data_cut.npy')

for row in a:
    for c in row:
        if c != 0:
            print(c)




# super_threshold_indices = a < 4
# print(np.count_nonzero(a))
# a[super_threshold_indices] = 0
# print(np.count_nonzero(a))
# np.save('data/data_cut.npy', a)