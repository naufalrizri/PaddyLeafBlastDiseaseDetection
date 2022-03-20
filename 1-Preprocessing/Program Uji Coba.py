# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 07:06:45 2020

@author: Naufal Rizki
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import imutils
import Cropping
from tqdm import tqdm

# Cek keberadaan file dalam sebuah folder
count = len([data for data in os.listdir(r"D:\Deep Learning\Blas\BLAST-YANMAR\Fix Used Image\20200217-Blas Sukabumi-Fitri") 
             if data.endswith(".jpg") and os.path.isfile(os.path.join(r"D:\Deep Learning\Blas\BLAST-YANMAR\Fix Used Image\20200217-Blas Sukabumi-Fitri", data))])

print("data gambar berjumlah :" + str(count) + " berekstensi(.jpg)")

# Memberikan salah satu contoh gambar sebelum dan sesudah di proses
gambar = cv.imread("D:\\Deep Learning\\Blas\\BLAST-YANMAR\\Fix Used Image\\20200217-Blas Sukabumi-Fitri\\" + "20200217_140300_1.jpg")
gambar = cv.cvtColor(gambar, cv.COLOR_BGR2RGB)
rotasi_gambar = imutils.rotate_bound(gambar, 45)
ukuran_crop = Cropping.largest_rotated_rect(gambar.shape[0], gambar.shape[1], 45)
crop_gambar = Cropping.crop_around_center(rotasi_gambar, ukuran_crop[0], ukuran_crop[1]) # angka ini didapat dari memotong ukuran gambar terkecil pada sudut 45 derajat maksimum
print("ukuran gambar sebelum dirotasi: " + str(gambar.shape))
print("ukuran gambar yang sudah dirotasi: " + str(rotasi_gambar.shape))
print("ukuran gambar setelah dicrop:" + str(crop_gambar.shape))

# Tampilkan Gambar
plt.grid(False)
fig = plt.figure();
ax = fig.add_subplot(1, 3, 1)
plot_gambar = plt.imshow(gambar)
ax.set_title("Gambar asli")
ax = fig.add_subplot(1, 3, 2)
plot_gambar = plt.imshow(rotasi_gambar)
ax.set_title("Gambar setelah dirotasi 45 derajat:")
ax = fig.add_subplot(1, 3, 3)
plot_gambar = plt.imshow(crop_gambar)
ax.set_title("Gambar setelah dirotasi dan dicrop:")