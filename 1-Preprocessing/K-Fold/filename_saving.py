# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:43:23 2021

@author: Naufal Rizki
"""

import os
import numpy as np
from random import shuffle

### Saving Filename

base_folder = r'D:\Deep Learning\Preprocessed_Classification'

filenames=[]
file_in_folder=[]

i=0

for folder in os.listdir(base_folder):
    i=0
    for files in os.listdir(os.path.join(base_folder, folder)):
        print(os.path.join(folder, files))
        filenames = np.append(filenames, os.path.join(folder, files))
        i += 1
    
    file_in_folder = np.append(file_in_folder, np.array([i]))
        
total_file = len(filenames)

np.savez(r'D:\Deep Learning\filenames_Preprocessed_Classified.npz', filenames=filenames, total=total_file, total_file_each=file_in_folder)