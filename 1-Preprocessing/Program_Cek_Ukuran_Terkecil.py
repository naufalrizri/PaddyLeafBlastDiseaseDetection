# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:13:37 2020

@author: Naufal Rizki
"""


import os
import numpy as np
import cv2 as cv
import Cropping
import pandas as pd
i = 1

path = r"D:\Deep Learning\BLAST-YANMAR-SOURCE-HPT\Fix Used Image"

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
      print(folder)
      print("=======================\n\n")
      nama_file = np.empty([1, 1])
      dimensi_file = np.empty([1, 3])
      ukuran_digunakan = np.empty([1, 2])      
      
      for files in os.listdir(os.path.join(path, folder)):
          
        if files.endswith(".jpg"):
          img = cv.imread(os.path.join(folder_path, files))
          ukuranGambar = np.reshape(np.array(img.shape), (1, 3))
          ukuranTerkecil = Cropping.largest_rotated_rect(ukuranGambar[:, 0], ukuranGambar[:, 1], 45)
          ukuranTerkecil = np.reshape(ukuranTerkecil, (1, 2))
          
          nama_file = np.append(nama_file, str(files))
          dimensi_file = np.append(dimensi_file, ukuranGambar, axis=0)
          ukuran_digunakan = np.append(ukuran_digunakan, ukuranTerkecil, axis=0)
          
        else:
          #print(files + "\t XXXXXX")
          pass
    
      nama_file = np.delete(nama_file, 0, axis=0)
      dimensi_file = np.delete(dimensi_file, 0, axis=0)
      ukuran_digunakan = np.delete(ukuran_digunakan, 0, axis=0)
      
      df = pd.DataFrame({
              'nama_file' : nama_file,
              'panjang' : dimensi_file[:, 0],
              'lebar' : dimensi_file[:, 1],
              'channel' : dimensi_file[:, 2],
              'panjang_terolah' : ukuran_digunakan[:, 0],
              'lebar_terolah' : ukuran_digunakan[:, 1],
              'selisih' : abs(ukuran_digunakan[:, 0] - ukuran_digunakan[:, 1])
          })
      
      df_1 = df.sort_values(by=['selisih'], ascending=False)
      df_2 = df.sort_values(by=['panjang_terolah'])
      df_3 = df.sort_values(by=['lebar_terolah'])
      
      writer = pd.ExcelWriter('rekap'+ str(i) + '_v1.1.xlsx', engine='xlsxwriter')
      
      i += 1
      
      df.to_excel(writer, sheet_name='raw')
      df_1.to_excel(writer, sheet_name='sort_by_difference')
      df_2.to_excel(writer, sheet_name='sort_by_panjang')
      df_3.to_excel(writer, sheet_name='sort_by_lebar')
      
      writer.save()
      
      del nama_file, dimensi_file, ukuran_digunakan, img, ukuranGambar, ukuranTerkecil, df, df_1, df_2, df_3, writer