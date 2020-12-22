from som import SOM
from scipy import sparse
import csv
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import MinMaxScaler
import utilities

if __name__ == '__main__':  
    # if this file exists, the SOM is already trained...
    if os.path.isfile('clusters_5x5.npy'):
        clusters = np.load('clusters_5x5.npy', allow_pickle=True)
        
        # ... so let's exports all images.
        utilities.save_clusters_images(clusters= clusters)

    else: 
        # Define the map shape.
        map_height = 5
        map_width = 5

        # initialize SOM 
        alpha_start = 0.6 # default value = 0.6
        seed = 2233 # default value = 2233
        nn = SOM(map_height, map_width, alpha_start=alpha_start, seed=seed)
        
        # load data 
        data = sparse.load_npz('sparse_B.npz').toarray()

        # initialize cluster matrix (shape: [#pixel, #cluster])
        clusters = np.zeros([data.shape[0], map_height*map_width], dtype=np.uint8)

        # train SOM. epochs = 0 means all dataset
        nn.fit(data=data, epochs=0, decay='hill')

        with open('csv_pixel-cluster.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            for idx, row in enumerate(data):
                row_winner, col_winner = nn.winner(row) # <--- get class and class coordinates --.
                cluster = 5 * row_winner + col_winner   # <--------------------------------------'
                csvwriter.writerow([idx, cluster])

                # this value is used to export cluster images
                clusters[idx][cluster] = 255

        np.save('clusters_5x5.npy', clusters)
        # saving SOM
        nn.save('som.p')