# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:43:24 2021

@author: Naufal Rizki
"""

### Making dataset with train=0.8 validation=0.1 and test=0.1

import os, shutil
import numpy as np
from random import shuffle

base_folder = r'D:\Deep Learning\Preprocessed_Classification_Revision'

target_folder = r'D:\Deep Learning\Dataset_Revision'

dataset_folder = os.listdir(target_folder) # array train, validation

# filename contains 'filenames', 'total', 'total_file_each'
data = np.load(r'D:\Deep Learning\filenames_Preprocessed_Classified_Revision.npz')
filenames = data['filenames']
total_file = int(data['total'])
total_classification = data['total_file_each']
folder_name = ['0-Tanpa_Gejala', "1-Gejala_Awal", "2-Gejala_Moderate", "3-Gejala_Lanjut", "4-Gejala_Lain"]


"""
    DATASET
    dataset is a folder with 2 subfolder Train, and Validation
    dividing the data with 80% Training, and 20% Validation
    each partition in training and validation have 5 class,
    so each class will divide into 80% for training and 20 % for testing
"""

# Copy file function
def copy_to_folder(base_folder, target_folder, file):
    src = os.path.join(base_folder, file)
    dst = os.path.join(target_folder, file)
    shutil.copyfile(src, dst)

folder_0 = [x for x in filenames if int(x[0]) == 0]
folder_1 = [x for x in filenames if int(x[0]) == 1]
folder_2 = [x for x in filenames if int(x[0]) == 2]
folder_3 = [x for x in filenames if int(x[0]) == 3]
folder_4 = [x for x in filenames if int(x[0]) == 4]

training_set = [3456, 3341, 14227, 5472, 346] # 26784, contains folder_0 to folder_4
validation_set = [864, 835, 3557, 1368, 86] # 6696, contains folder_0 to folder_4
# validation_set = [374, 418, 1771, 684, 43] # 3348, contains folder_0 to folder_4
# test_set = [374, 417, 1772, 684, 43] # 3348, contains folder_0 to folder_4

# shuffle all files in folder
for j in range(5):
    shuffle(eval("folder_" + str(j)))

# 80% training, 20% validation

for i in range(5):
    folder = eval("folder_" + str(i))
    training_index = training_set[i]
    print(training_index)
    validation_index = validation_set[i] + training_index
    print(validation_index)
    # test_index = test_set[i] + validation_index
    # print(test_index)
    
    for file in folder[0:training_index]:
        # copy all file to training folder
        copy_to_folder(base_folder, os.path.join(target_folder, "Train"), file)
    
    for file in folder[training_index:validation_index]:
        copy_to_folder(base_folder, os.path.join(target_folder, "Validation"), file)
    
    # for file in folder[validation_index:test_index]:
    #     copy_to_folder(base_folder, os.path.join(target_folder, "Test"), file)
    
    # sanity check
    print("folder ==> {}".format(i))
    print("total training: {}".format(len(os.listdir(os.path.join(r'D:\Deep Learning\Dataset_Revision\Train', folder_name[i])))))
    print("total validation: {}".format(len(os.listdir(os.path.join(r'D:\Deep Learning\Dataset_Revision\Validation', folder_name[i])))))
    # print("total test: {}".format(len(os.listdir(os.path.join(r'D:\Deep Learning\Blas\Dataset_v2_80_10_10\Test', folder_name[i])))))