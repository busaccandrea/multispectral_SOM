from numpy.lib.npyio import load
from som import SOM
from scipy import sparse
import csv
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import MinMaxScaler
import utilities
import reduce_dim_spectra as reduce_dim
import processing_data
from compare import Comparison

if __name__ == '__main__':  
    # if this file exists, the SOM is already trained...
    # Define the map shape.
    map_height = 2
    map_width = 4
######################################### SET THIS VAR BEFORE LAUNCH ########################################################    
    reduction_method = 'pca'                                                                                                #
#############################################################################################################################
    # for NNMF
    n_components = 10
    smoothing = 5


    if os.path.isfile('data/' + reduction_method + '/' + 'clusters_' + str(n_components) + '_' + str(smoothing) + '_' + str(map_height) + 'x' + str(map_width) +'.npy'):
        clusters_perc = np.load('data/' + reduction_method + '/' + 'clusters_' + str(n_components) + '_' + str(smoothing) + '_' + str(map_height) + 'x' + str(map_width) +'.npy', allow_pickle=True)

        # ... so let's exports all images.
        print('Saving images...')
        utilities.save_clusters_images(clusters_perc=clusters_perc, basename=reduction_method + '/')
        print('Done. Saving spectras...')
        utilities.get_clusters_spectras(clusters_perc, filename= reduction_method, write_in_file=True)
        print('Done.')

    else:
        if reduction_method == 'nmf':
            # load data 
            datafilename = 'data/' + reduction_method + '/W_' + str(n_components) + '_comp_smth_' + str(smoothing) + '.npy'
        else:
            datafilename = 'data/' + reduction_method + '/data_' + str(n_components) + '_comp' + '.npy'


        if os.path.isfile(datafilename):
            data = np.load(datafilename)
            print('Loaded data.')
        else: 
            B = np.load('data/B.npy')
            print('Loaded B matrix. Reducing data...')
            if reduction_method == 'nmf':
                data = reduce_dim.nnmf_based(data=B, smoothing=smoothing, n_components_=n_components, write_to_file=True)
            else:
                data = reduce_dim.pca_based(data=B, n_components_=n_components, write_to_file=True)
            print("Done and saved.")


        data = data[:, 0:4]
        print(data.shape)

        # initialize cluster matrix (shape: [#pixel, #cluster])
        clusters = np.zeros([data.shape[0], map_height*map_width], dtype=np.uint8)
        percentages = np.zeros([data.shape[0], map_height*map_width], dtype=np.float32)
        
        # initialize SOM 
        alpha_start = 1 # default value = 0.6
        seed = 2233 # default value = 2233
        nn = SOM(map_height, map_width, alpha_start=alpha_start, seed=seed)
        som_file = 'data/checkpoints/som_' + reduction_method + str(n_components) + '_' + str(smoothing) + '_' + str(map_height) + 'x' + str(map_width) +'.p'
        if os.path.isfile(som_file):
            nn.load(filename=som_file)
            print('Loaded som.')
        else:
            print('training som...')
            # train SOM. epochs = 0 means all dataset
            nn.fit(data=data, epochs=0, decay='hill')
            nn.save(som_file)

        print('\nclusters...')
        for idx, row in enumerate(data):
            # row_winner, col_winner = nn.winner(row) # <--- get class and class coordinates ----------.
            # cluster = map_width * row_winner + col_winner   # <--------------------------------------'
            distances = np.sum((nn.map - row) ** 2, axis=2)
            percentages[idx] = processing_data.data_clustering(distances)
            clusters[idx] = np.uint8(255 * percentages[idx])
        # np.save('clusters_' + reduction_method + '_' + str(map_height) + 'x' + str(map_width)  +'.npy', clusters)
        np.save('data/som/' + reduction_method + '/' + 'clusters_' + str(n_components) + '_' + str(smoothing) + '_' + str(map_height) + 'x' + str(map_width) +'.npy', clusters)

        # # ... so let's exports all images.
        print('Saving images...')
        utilities.save_clusters_images(clusters_perc=clusters, basename='som/' + reduction_method + '/')
        print('Done. Saving spectras...')
        utilities.get_clusters_spectras(clusters, filename= 'som/' + reduction_method + '/', write_in_file=True)
        print('Done.')


        # comparison metrics
        comparison = Comparison(data=clusters)
        comparison.compute_all(verbose=True)