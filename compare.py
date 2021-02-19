from PIL import Image
import utilities
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import glob
from os.path import basename
from matplotlib import pyplot as plt


class Comparison(object):
    """ Description """
    def __init__(self, data):
        self.data = data
        self.chem_el_files = glob.glob('data/DOggionoGiulia/PNG-ed/*.png')
        self.M = utilities.get_matrix_from_chem_el_tiffs(self.chem_el_files)
        
        # this line are used to convert tiff images in png images and/or 
        # self._normalize_M()

        # mse results is a matrix where rows are the clustering results and cols are elemental samples. mse_results[i][j] is mse between each elemental 
        # sample and each SOM result
        self.mses = np.zeros((data.shape[1], len(self.chem_el_files)))
        # same of mse
        self.ssims = np.zeros((data.shape[1], len(self.chem_el_files)))
        #same of mse
        self.pearsons = np.zeros((data.shape[1], len(self.chem_el_files)))
        self.spearmans = np.zeros((data.shape[1], len(self.chem_el_files)))
    
    def _normalize_M(self):
        max = np.max(self.M)
        self.M = self.M.clip(min=0)
        for i, _ in enumerate(self.M):
            self.M[i] = ((self.M[i])/max)*255

        # these lines are uset to convert tiff images in png images
        # for j in range(0, self.M.shape[1]):
        #     im = self.M[:, j].reshape((418, 418))
        #     img = Image.fromarray(im)
        #     img = img.convert("L")
        #     img.save(self.chem_el_files[j] + '.PNG')

    def compute_all(self, verbose = True):
        """ 
        Compute all metrics in this class between SOM results and D'Oggiono Giulia files at data/DOggionoGiulia
        """
        for j, f in enumerate(self.chem_el_files):
            arr = self.M[:, j]
            for i in range(self.data.shape[1]):
                my_arr = self.data[:, i]
                self.ssims[i, j] = ssim(my_arr.astype(np.float64), arr)
                self.mses[i, j] = mse(my_arr, arr)
                (corr, p_value) = pearsonr(my_arr, arr)
                self.pearsons[i, j] = corr
                (corr, p_value) =  spearmanr(my_arr, arr)
                self.spearmans[i, j] = corr

        if verbose:
            for cluster in range(self.mses.shape[0]):
                fig, axs = plt.subplots(2, 2)
                fig.suptitle('Comparing ' + str(cluster) + ' with tiffs')
                # pearson
                axs[0, 0].plot(self.pearsons[cluster], 'tab:blue')
                axs[0, 0].set_ylim([-1, 1])
                axs[0, 0].set_title('pearson')

                # spearman
                axs[0, 1].plot(self.spearmans[cluster], 'tab:orange')
                axs[0, 1].set_ylim([-1, 1])
                axs[0, 1].set_title('spearman')

                # mse
                axs[1, 0].plot(self.mses[cluster], 'tab:green')
                axs[1, 0].set_title('mse')

                # ssim
                axs[1, 1].plot(self.ssims[cluster], 'tab:red')
                axs[1, 1].set_ylim([0, 1])
                axs[1, 1].set_title('ssim')
                
                for ax in axs.flat:
                    ax.set(xlabel='tiff images', ylabel='metric value')
                
                # show plot in fullscreen
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()

                plt.show()


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

# clusters = np.load('C:/Users/apost/Pavel Folder/VS Code folder/multispectral_SOM/data/cmeans/nmf/clusters_8comp_15smth_8groups.npy')
# comp = Comparison(data=clusters)
# comp.compute_all(verbose=True)