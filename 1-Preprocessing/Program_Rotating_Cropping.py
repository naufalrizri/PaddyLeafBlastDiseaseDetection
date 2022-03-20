# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 08:45:00 2021

@author: Naufal Rizki
"""

import numpy as np
import cv2 as cv
import imutils
import Cropping
from tqdm import tqdm
import os

base_folder = r'D:\Deep Learning\Raw_Classification_Revision'

target_folder = r'D:\Deep Learning\Preprocessed_Classification_Revision'

# Membuat folder output

# folder1 = os.path.join(target_folder, "1-Gejala_Awal")

# folder2 = os.path.join(target_folder, "2-Gejala_Moderate")

# folder3 = os.path.join(target_folder, "3-Gejala_Lanjut")

# folder4 = os.path.join(target_folder, "4-Gejala_Lain")

folder0 = os.path.join(target_folder, "0-Tanpa_Gejala")

# if not os.path.exists(folder1) or not os.path.exists(folder2) or not os.path.exists(folder3) or not os.path.exists(folder4):
    # os.mkdir(folder1)
    # os.mkdir(folder2)
    # os.mkdir(folder3)
    # os.mkdir(folder4)
    
if not os.path.exists(folder0):    
    os.mkdir(folder0)
    

for folder in os.listdir(base_folder):
    fnames = os.path.join(base_folder, folder)
    print('\n', folder, '\n', '============================')
    
    # Cek isi Per Folder
    count = len([data for data in os.listdir(fnames) if data.endswith(".jpg") 
    and os.path.isfile(os.path.join(fnames, data))])
    
    pbar = tqdm(total=count)
    if os.path.isdir(os.path.join(base_folder, folder)):    
        for fname in os.listdir(fnames):
            pic = cv.imread(os.path.join(fnames, fname))
            print('\n', fname)
            
            for angle in np.arange(0, 360, 5):
                rotated = imutils.rotate_bound(pic, angle)
                center_crop = Cropping.crop_around_center(rotated, 779, 779)
                cv.imwrite(target_folder +"\\" + folder + '\\' + fname[:-4] + "_{}.jpg".format(angle), center_crop)
                
            pbar.update(1)
    pbar.close()