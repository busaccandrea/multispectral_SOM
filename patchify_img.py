from PIL import Image
from sklearn.feature_extraction import image as ski
import os
import numpy as np
from skimage import io, transform
from os.path import split
from utilities import check_existing_folder
import csv

""" this method splits two imgfiles into N patches 64x64 
    User can set the patch dimension.
"""

def patchify(rgb_file, pbl_file, patch_size_=64):
    # read images
    RGB = io.imread(rgb_file)
    PBL = io.imread(pbl_file)
    # getting file path
    base, _ = split(rgb_file)
    
    # checking if output folders exist
    check_existing_folder(base + '/RGBPatches/')
    check_existing_folder(base + '/PbLPatches/')

    # patchify images
    RGB_patches = ski.extract_patches_2d(RGB, patch_size=(patch_size_, patch_size_)) # shape=(N, 64, 64, 3)
    PBL_patches = ski.extract_patches_2d(PBL, patch_size=(patch_size_, patch_size_)) # shape=(N, 64, 64)
    
    with open(base + '/dataset.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        # for through number of patches
        for patch_index in range(0, RGB_patches.shape[0]):
            if int(patch_index/RGB.shape[1]) % 63 == 0 and patch_index % 60 == 0:
                rgbname = base + '/RGBPatches/rgb_' + str(patch_index) +'.png'
                im = Image.fromarray(RGB_patches[patch_index].astype(np.uint8))
                im.save(rgbname)
                
                pblname = base + '/PbLPatches/pbl_' + str(patch_index) +'.png'
                im = Image.fromarray(PBL_patches[patch_index].astype(np.uint8))
                im.save(pblname)

                # # write a row in csv: rgbname,pblname
                csv_writer.writerow([rgbname,pblname])
                # print(patch_index)


# patchify(rgb_file='data/RGB2Gray/Andrea/PaoloMolaro/A2021_R01_VIS.tiff', pbl_file='data/RGB2Gray/Andrea/PaoloMolaro/A2021_R01_Pb_L.tiff')