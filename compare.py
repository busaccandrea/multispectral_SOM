import utilities
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import glob
from os.path import basename


class Comparison(object):
    """ Description """
    def __init__(self, data):
        self.data = data
        self.chem_el_files = glob.glob('data/DOggionoGiulia/*.tiff')
        self.M = utilities.get_matrix_from_chem_el_tiffs(self.chem_el_files)
        # mse results is a matrix where rows are the SOM result and cols are elemental samples. mse_results[i][j] is mse between each elemental 
        # sample and each SOM result
        self.mses = np.zeros((data.shape[1], len(self.chem_el_files)))
        # same of mse
        self.ssims = np.zeros((data.shape[1], len(self.chem_el_files)))
        #same of mse
        self.pearsons = np.zeros((data.shape[1], len(self.chem_el_files), 2))
        self.spearmans = np.zeros((data.shape[1], len(self.chem_el_files), 2))
    

    def compute_all(self, verbose = False):
        """ 
        Compute all metrics in this class between SOM results and D'Oggiono Giulia files at data/DOggionoGiulia
        """
        for j, f in enumerate(self.chem_el_files):
            arr = self.M[:, j]
            for i in range(self.data.shape[1]):
                my_arr = self.data[:, i]
                self.ssims[i, j] = ssim(my_arr.astype(np.float32), arr)
                self.mses[i, j] = mse(my_arr, arr)
                self.pearsons[i, j] = pearsonr(my_arr, arr)
                self.spearmans[i, j] = spearmanr(my_arr, arr)
                if verbose:
                    print('Comparage between', basename(f), 'and', i)
                    print('\t\tssim:', round(self.ssims[i, j], 4))
                    print('\t\tpearson:', round(self.pearsons[i, j, 0], 4))
                    print('\t\tMSE:', round(self.mses[i,j], 4))
                    print('\t\tSpearman:', round(self.spearmans[i, j, 0], 4))


    def pearson_correlation(self, verbose = False):
        """ 
        Compute Pearson Correlation between SOM results and D'Oggiono Giulia files at data/DOggionoGiulia
        """
        for j, f in enumerate(self.chem_el_files):
            arr = self.M[:, j]
            for i in range(self.data.shape[1]):
                my_arr = self.data[:, i]
                self.pearsons[i, j] = pearsonr(my_arr, arr)
                if verbose:
                    print('pearson between', basename(f), 'and', i, ':', round(self.pearsons[i, j, 0], 4))
    
    
    def mse_(self, verbose = False):
        """ 
        Compute Mean Square Error between SOM results and D'Oggiono Giulia files at data/DOggionoGiulia
        """
        for j, f in enumerate(self.chem_el_files):
            arr = self.M[:, j]
            for i in range(self.data.shape[1]):
                my_arr = self.data[:, i]
                self.mses[i, j] = mse(my_arr, arr)
                
                if verbose:
                    print('MSE between', basename(f), 'and', i, ':', round(self.mses[i,j], 4))
                            

    def ssim_(self, verbose = False):
        """ 
        Compute Structure Similarity between SOM results and D'Oggiono Giulia files at data/DOggionoGiulia
        """
        for j, f in enumerate(self.chem_el_files):
            arr = self.M[:, j]
            for i in range(self.data.shape[1]):
                my_arr = self.data[:, i]
                self.ssims[i, j] = ssim(my_arr, arr)
                if verbose:
                    print('SSIM between', basename(f), 'and', i, ':', round(self.ssims[i, j], 4))                        


    def spearman(self, verbose = False):
        """ 
        Compute Structure Similarity between SOM results and D'Oggiono Giulia files at data/DOggionoGiulia
        """
        for j, f in enumerate(self.chem_el_files):
            arr = self.M[:, j]
            for i in range(self.data.shape[1]):
                my_arr = self.data[:, i]
                self.spearmans[i, j] = spearmanr(my_arr, arr)
                if verbose:
                    print('Spearman between', basename(f), 'and', i, ':', round(self.spearmans[i, j, 0], 4))
        



clusters = np.load('results/campioni/som2x3/nmf/6comp/smth 15/clusters_6_15_2x3.npy')
comp = Comparison(data=clusters)
# compare.pearson_correlation(verbose=True)
# compare.mse_(verbose=True)
max = np.max(comp.M)
min = np.min(comp.M)
for i, _ in enumerate(comp.M):
    comp.M[i] = ((comp.M[i] - min)/(max - min))*255


comp.compute_all(verbose=True)