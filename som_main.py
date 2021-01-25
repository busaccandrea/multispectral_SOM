from som import SOM
from scipy import sparse
import csv
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import MinMaxScaler
import utilities
import reduce_dim_spectra as reduce_dim

if __name__ == '__main__':  
    # if this file exists, the SOM is already trained...
    # Define the map shape.
    map_height = 4
    map_width = 4
######################################### SET THIS VAR BEFORE LAUNCH ########################################################    
    reduction_method = 'nmf'                                                                                                #
#############################################################################################################################
    if os.path.isfile('clusters_' + reduction_method + '_' + str(map_height) + 'x' + str(map_width) + '.npy'):
        clusters = np.load('clusters_' + reduction_method + '_' + str(map_height) + 'x' + str(map_width) + '.npy', allow_pickle=True)
        
        # ... so let's exports all images.
        utilities.save_clusters_images(clusters=clusters, base_filename=reduction_method)

    else:
        # initialize SOM 
        alpha_start = 0.6 # default value = 0.6
        seed = 2233 # default value = 2233
        nn = SOM(map_height, map_width, alpha_start=alpha_start, seed=seed)
        
        # load data 
        # data = sparse.load_npz('sparse_B.npz').toarray()
        datafilename = 'data_' + reduction_method + '_8.npy'
        data = np.load(datafilename)

        # initialize cluster matrix (shape: [#pixel, #cluster])
        clusters = np.zeros([data.shape[0], map_height*map_width], dtype=bool)
        # train SOM. epochs = 0 means all dataset
        nn.fit(data=data, epochs=0, decay='hill')

        print('clusters...')
        # with open('pixel-cluster.csv', 'w', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile, delimiter=' ')
        #     for idx, row in enumerate(data):
        #         # debugging
        #         # print('idx=', idx)
        #         row_winner, col_winner = nn.winner(row) # <--- get class and class coordinates ----------.
        #         cluster = map_width * row_winner + col_winner   # <--------------------------------------'
        #         csvwriter.writerow([idx, cluster])

        #         # debugging
        #         # print('cluster = ', cluster)

        #         # this value is used to export cluster images
        #         # clusters[idx][cluster] = 255    # cluster - 0 for 0-indexing dataset-label
        #         clusters[idx][cluster-1] = 255  # cluster - 1 for 1-indexing dataset-label
        for idx, row in enumerate(data):
            row_winner, col_winner = nn.winner(row) # <--- get class and class coordinates ----------.
            cluster = map_width * row_winner + col_winner   # <--------------------------------------'

            clusters[idx][cluster-1] = 1  # cluster - 1 for 1-indexing dataset-label


        np.save('clusters_' + reduction_method + '_' + str(map_height) + 'x' + str(map_width)  +'.npy', clusters)
        # saving SOM
        nn.save('som.p')