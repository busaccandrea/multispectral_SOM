import scipy.io as scio
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# # per caricare un file .mat usare la libreria di scipy.io
mat_file = scio.loadmat('./B.mat')
B_ = mat_file["B"]
B = np.zeros(B_.shape)

# spectra = B_[0]





# # normalizzazione
# scaler = MinMaxScaler(feature_range=(0, 255))
# B = scaler.fit_transform(B)

# #salvataggio in matrice sparsa
# sparse_B = sparse.csr_matrix(B)
# sparse.save_npz("sparse_B", sparse_B, compressed=True)


# # # per smoothing
# smth = 5 # dimensione media mobile
# half_smth = int(smth/2)
# smoothed_data = np.zeros(B.shape, dtype=float)

# for row in range(0, B.shape[0]):
#     # utilizzo la convoluzione per determinare in modo efficiente la media mobile sulle righe di B
#     smoothed_data[row] = np.convolve(B[row], np.ones(smth), 'same') / smth
# sparse_SB = sparse.csr_matrix(smoothed_data)
# sparse.save_npz('B_smth' + str(smth), sparse_SB, compressed=True)


