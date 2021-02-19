from numpy.lib.npyio import load
from fcmeans import FCM
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

######################################### SET THIS VAR BEFORE LAUNCH ########################################################    
    reduction_method = 'pca'                                                                                                #
#############################################################################################################################
    # for NNMF
    n_components = 10
    smoothing = 0
    n_clusters = 10

    if reduction_method == 'pca':
        # load data 
        datafilename = 'data/' + reduction_method + '/W_' + str(n_components) + '_comp_smth_' + str(smoothing) + '.npy'
    else:
        datafilename = 'data/' + reduction_method + '/data_' + str(n_components) + '_comp_smth_' + str(smoothing) + '.npy'


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

    # initialize cluster matrix (shape: [#pixel, #cluster])
    clusters = np.zeros([data.shape[0], n_clusters], dtype=np.uint8)
    percentages = np.zeros([data.shape[0], n_clusters], dtype=np.float32)
    
    # initialize SOM 
    seed = 2233 # default value = 2233
    cmeans = FCM(n_clusters=n_clusters, random_state=seed)

    print('training cmeans...')
    # train SOM. epochs = 0 means all dataset
    cmeans.fit(data)

    print('\nclusters...')
    for idx, row in enumerate(data):
        distances = cmeans.predict(row)
        percentages[idx] = processing_data.data_clustering_cmeans(distances)
        clusters[idx] = np.uint8(255 * percentages[idx])
    np.save('data/cmeans/' + reduction_method + '/clusters_' + str(n_components) + 'comp_' + str(smoothing) + 'smth_' + str(n_clusters) + 'groups.npy', clusters)

    # ... so let's exports all images.
    print('Saving images...')
    utilities.save_clusters_images(clusters_perc=clusters, basename='cmeans/' + reduction_method + '/')
    print('Done. Saving spectras...')
    utilities.get_clusters_spectras(clusters, filename= 'cmeans/' + reduction_method + '/', write_in_file=True)
    print('Done.')

    # comparison metrics
    comparison = Comparison(data=clusters)
    comparison.compute_all()
