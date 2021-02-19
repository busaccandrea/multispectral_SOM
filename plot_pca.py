import matplotlib.pyplot as plt
import numpy as np

pcafile = 'variance_ratio_10_comp.npy'
f = np.load('data/pca/' + pcafile)

plt.plot(f)
plt.show()