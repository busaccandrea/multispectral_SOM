from Pavencoder import load_model_for_inference
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
import torch

if __name__ == '__main__':  
    # if this file exists, the SOM is already trained...
    # Define the map shape.
    map_height = 2
    map_width = 4
# ######################################### SET THIS VAR BEFORE LAUNCH ########################################################    
#     reduction_method = 'pca'                                                                                                #
# #############################################################################################################################
#     # for NNMF
#     n_components = 10
#     smoothing = 5


    if os.path.isfile('data/e.npy'):
        clusters_perc = np.load('datanpy', allow_pickle=True)

    else:
        data = np.load('data/codes.npy')
        # initialize cluster matrix (shape: [#pixel, #cluster])
        clusters = np.zeros([data.shape[0], map_height*map_width], dtype=np.uint8)
        percentages = np.zeros([data.shape[0], map_height*map_width], dtype=np.float32)
        
        # initialize SOM 
        alpha_start = 1 # default value = 0.6
        seed = 2233 # default value = 2233
        nn = SOM(map_height, map_width, alpha_start=alpha_start, seed=seed)
        som_file = 'data/checkpoints/som_pavenc_' + str(map_height) + 'x' + str(map_width) + '.p'
        if os.path.isfile(som_file):
            nn.load(filename=som_file)
            print('Loaded som.')
        else:
            print('training som...')
            # train SOM. epochs = 0 means all dataset
            nn.fit(data=data, epochs=0, decay='hill')
            print('save')
            nn.save(som_file)
        

        model_filename = 'data/pavencoder/' +  'model.pt'
        checkpoint_filename = 'data/pavencoder/' + 'model_checkpoint.pt'
        if os.path.isfile(model_filename) and os.path.isfile(checkpoint_filename):
            print('Model and checkpoint files found. Loading.')
            [model, optimizer, checkpoint] = load_model_for_inference(model_filename, checkpoint_filename)
            model.double()

        print('\nclusters...')
        for idx, row in enumerate(data):
            # row_winner, col_winner = nn.winner(row) # <--- get class and class coordinates ----------.
            # cluster = map_width * row_winner + col_winner   # <--------------------------------------'
            distances = np.sum((nn.map - row) ** 2, axis=2)
            percentages[idx] = processing_data.data_clustering(distances)
            clusters[idx] = np.uint8(255 * percentages[idx])
        
        utilities.check_existing_folder('data/som/pavencoder/')
        # np.save('clusters_' + reduction_method + '_' + str(map_height) + 'x' + str(map_width)  +'.npy', clusters)
        np.save('data/som/pavencoder/clusters_' + str(map_height) + 'x' + str(map_width) +'.npy', clusters)

        # # ... so let's exports all images.
        print('Saving images...')
        utilities.save_clusters_images(clusters_perc=clusters, basename='som/pavencoder/')
        print('Done. Saving spectras...')
        utilities.get_clusters_spectras(clusters, filename= 'som/pavencoder/', write_in_file=True)
        print('Done.')


        # comparison metrics
        comparison = Comparison(data=clusters)
        comparison.compute_all(verbose=True)