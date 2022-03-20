# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 05:54:12 2020

@author: Naufal Rizki
"""

import numpy as np
import cv2 as cv
import os
import imutils
import Cropping
from tqdm import tqdm

"""
Program python untuk melakukan pemrosesan gambar berupa rotasi dan memotong gambar padi
"""

# Mengolah keseluruhan gambar

# Membuat folder output
os.mkdir("D:\\Deep Learning\\Blas\\Preprocessing\\Preprocessing1")

folder_target=r"D:\\Deep Learning\\Blas\\BLAST-YANMAR\\Choosed\\20200217-Blas Sukabumi-Fitri-Choosed\\"
nama_folder_baru = "Folder1@"
count = len([data for data in os.listdir(folder_target) if data.endswith(".jpg") 
and os.path.isfile(os.path.join(folder_target, data))])
pbar = tqdm(total=count)
for name in os.listdir(folder_target):

  if name.endswith(".jpg"):
    os.mkdir("D:\\Deep Learning\\Blas\\Preprocessing\\Preprocessing1\\" + nama_folder_baru + name)
      
    # Baca gambar
    gambar = cv.imread(folder_target + name)

    # Simpan gambar asli
    cv.imwrite(r"D:\\Deep Learning\\Blas\\Preprocessing\\Preprocessing1\\" + nama_folder_baru + name + "/" + "RAW.jpg", gambar)

    # Manipulasi Gambar
    # Rotasi
    for angle in np.arange(0, 360, 5):
      rotated = imutils.rotate_bound(gambar, angle)
      # Crop Gambar
      crop_tengah = Cropping.crop_around_center(rotated, 779, 779)
      # masukkan ke folder preprocessing
      cv.imwrite(r"D:\\Deep Learning\\Blas\\Preprocessing\\Preprocessing1\\" + nama_folder_baru + name + "/" + " rotated "+ str(angle) + " degree.jpg", crop_tengah)
      
  pbar.update(1)

pbar.close()
