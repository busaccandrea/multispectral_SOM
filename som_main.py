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
from matplotlib import pyplot as plt
from compare import Comparison

if __name__ == '__main__':  
    # if this file exists, the SOM is already trained...
    # Define the map shape.
    map_height = 3
    map_width = 3
######################################### SET THIS VAR BEFORE LAUNCH ########################################################    
    reduction_method = 'nmf'                                                                                                #
#############################################################################################################################
    # for NNMF
    n_components = 10
    smoothing = 5


    if os.path.isfile('data1/' + reduction_method + '/' + 'clusters_' + str(n_components) + '_' + str(smoothing) + '_' + str(map_height) + 'x' + str(map_width) +'.npy'):
        clusters_perc = np.load('data/' + reduction_method + '/' + 'clusters_' + str(n_components) + '_' + str(smoothing) + '_' + str(map_height) + 'x' + str(map_width) +'.npy', allow_pickle=True)

        # ... so let's exports all images.
        print('Saving images...')
        utilities.save_clusters_images(clusters_perc=clusters_perc, basename=reduction_method + '/')
        print('Done. Saving spectras...')
        utilities.get_clusters_spectras(clusters_perc, filename= reduction_method, write_in_file=True)
        print('Done.')

    else:
    #     if reduction_method == 'nmf':
    #         # load data 
    #         datafilename = 'data/' + reduction_method + '/W_' + str(n_components) + '_comp_smth_' + str(smoothing) + '.npy'
    #     else:
    #         datafilename = 'data/' + reduction_method + '/data_' + str(n_components) + '_comp' + '.npy'

        B = np.load('data/Edf20MS/data_nobg.npy')
        data = B.copy()
        # np.random.shuffle(data)
        # data[:, 0:50] = 0
        # data[:, 1200:-1] = 0
    #     np.save('data/data_cut.npy', B)

        if False:
        # if os.path.isfile(datafilename):
        #     data = np.load(datafilename)
            print('Loaded data.')
            
        else: 
            print('Loaded B matrix. Reducing data...')
            # data = np.load('data/Edf20MS/codes_from_matlab.npy')
            # data = B[:, 50 :1250]

            # if reduction_method == 'nmf':
            #     data = reduce_dim.nnmf_based(data=B, smoothing=smoothing, n_components_=n_components, write_to_file=True)
            # else:
            #     data = reduce_dim.pca_based(data=B, n_components_=n_components, write_to_file=True)
            print("Done and saved.")



        # initialize cluster matrix (shape: [#pixel, #cluster])
        clusters = np.zeros([data.shape[0], map_height*map_width], dtype=np.uint8)
        percentages = np.zeros([data.shape[0], map_height*map_width], dtype=np.float32)
        
        # initialize SOM 
        alpha_start = 0.6 # default value = 0.6 
        seed = 2233 # default value = 2233
        nn = SOM(map_height, map_width, alpha_start=alpha_start, seed=seed)
        som_file = 'data1/checkpoints/som_' + reduction_method + str(n_components) + '_' + str(smoothing) + '_' + str(map_height) + 'x' + str(map_width) +'.p'
        if os.path.isfile(som_file):
            nn.load(filename=som_file)
            print('Loaded som.')
        else:
            print('training som...')
            # train SOM. epochs = 0 means all dataset
            nn.initialize(data)
            nnmap = nn.map.copy()
            nn.fit(data=data, epochs=0, decay='hill')
            # nn.save(som_file)


        # check neurons' vector pre-fit and post-fit
        for row in range(0, nn.map.shape[0]):
            for col in range(0, nn.map.shape[1]):
                plt.figure
                plt.plot(nn.map[row,col], 'g', nnmap[row,col], 'r') #,data[1000],'r'
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
                plt.show()
        # quit()


        print('\nclusters...')
        for idx, row in enumerate(data):
            # row_winner, col_winner = nn.winner(row) # <--- get class and class coordinates ----------.
            # cluster = map_width * row_winner + col_winner   # <--------------------------------------'

            """ distances = np.sum((nn.map - row) ** 2, axis=2)
            percentages[idx] = processing_data.data_clustering(distances)
            ind_max = np.argmax(percentages[idx])
            percentages[idx] *= 0
            percentages[idx][ind_max] = 1 """
            winner = nn.winner(row)
            percentages[idx, winner] = 1
            clusters[idx] = np.uint8(255 * percentages[idx])
        # np.save('clusters_' + reduction_method + '_' + str(map_height) + 'x' + str(map_width)  +'.npy', clusters)
        np.save('data/som/' + reduction_method + '/' + 'clusters_' + str(n_components) + '_' + str(smoothing) + '_' + str(map_height) + 'x' + str(map_width) +'.npy', clusters)

        # # ... so let's exports all images.
        print('Saving images...')
        utilities.save_clusters_images(clusters_perc=clusters, basename='som/' + reduction_method + '/')
        print('Done. Saving spectras...')
        utilities.get_clusters_spectras(clusters, data=B, filename= 'som/' + reduction_method + '/', write_in_file=True)
        print('Done.')


        # comparison metrics
        comparison = Comparison(data=clusters)
        comparison.compute_all(verbose=True)