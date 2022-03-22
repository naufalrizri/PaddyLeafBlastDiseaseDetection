# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:31:05 2021

@author: Lenovo
"""

from tkinter import *
from PIL import ImageTk, Image
import tkinter.filedialog
import cv2
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


def DefineModel(weightPath=r"blast_saved_at_20.h5"):

    # for deep learning model
    import XCEPTION_MODEL

    model = XCEPTION_MODEL.make_model(
        input_shape=(700, 700) + (3,), num_classes=5)
    model.load_weights(weightPath)
    # print(model.summary())

    return model


def ImageToArray(path=None, dim=(700, 700)):

    # originalImage = cv2.imread(path)
    # resizedImage = cv2.resize(originalImage, dim)

    # image = np.expand_dims(resizedImage, axis=0)

    originalImage = Image.open(path)
    resizedImage = Image.Image.resize(originalImage, dim)
    imageArray = np.array(resizedImage)
    # plt.imshow(gambar)
    image = np.squeeze(imageArray)
    image = np.expand_dims(image, 0)

    return image


def PredictImage(imageArray, model=None):

    prediction = model.predict(imageArray)
    print(np.argmax(prediction, axis=1))

    return np.argmax(prediction, axis=1)


def VisualizeHeatmap(imageArray, layerName=None, model=None):

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(layerName)
        iterate = tf.keras.models.Model(
            [model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(imageArray)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(
        pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((heatmap.shape[1], heatmap.shape[2]))

    # plt.matshow(heatmap)
    # plt.show()
    return heatmap


def Visualize(imageArray, heatmap=None, INTENSITY=.2):
    
    image = imageArray[0]

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    image = heatmap * INTENSITY + image

    return image.astype('uint8')


def SelectImage(model):
    global panelA, panelB, w
    path = tkinter.filedialog.askopenfilename()

    # Pengolahan
    imageArray = ImageToArray(path=path)

    # Prediksi
    predictionValue = PredictImage(imageArray, model=model)

    if predictionValue == 0:
        prediction = 'Daun Tanaman Padi Tidak Berpenyakit' + \
            ' (' + str(predictionValue) + ')'
    elif predictionValue == 1:
        prediction = 'Terindikasi Gejala Penyakit Blas Fase Awal' + \
            ' (' + str(predictionValue) + ')'
    elif predictionValue == 2:
        prediction = 'Terindikasi Gejala Penyakit Blas Fase Pertengahan' + \
            ' (' + str(predictionValue) + ')'
    elif predictionValue == 3:
        prediction = 'Terindikasi Gejala Penyakit Blas Fase Lanjutan' + \
            ' (' + str(predictionValue) + ')'
    else:
        prediction = 'Terindikasi Gejala Penyakit Bukan Blas' + \
            ' (' + str(predictionValue) + ')'

    # Visualisasi
    # Heatmap
    heatmap = VisualizeHeatmap(imageArray, layerName=str(
        model.layers[-8].name), model=model)

    # Gabungan dengan Gambar
    visualizeGradCAM = Visualize(imageArray, heatmap=heatmap)

    # Label
    w = Label(text=prediction)
    w.pack()

    # callback array to image
    originalImage = cv2.cvtColor(imageArray[0], cv2.COLOR_BGR2RGB)
    visualImage = cv2.cvtColor(visualizeGradCAM, cv2.COLOR_BGR2RGB)
    originalImage = cv2.resize(
        imageArray[0], (200, 200), interpolation=cv2.INTER_AREA)
    visualImage = cv2.resize(visualizeGradCAM, (200, 200),
                             interpolation=cv2.INTER_AREA)
    originalImage = Image.fromarray(originalImage)
    visualImage = Image.fromarray(visualImage)
    originalTkImage = ImageTk.PhotoImage(originalImage)
    visualTkImage = ImageTk.PhotoImage(visualImage)

    # Show the image in Tkinter
    if panelA is None or panelB is None:
        panelA = Label(image=originalTkImage)
        panelA.image = originalTkImage
        panelA.pack(side="left", padx=10, pady=10)
        panelB = Label(image=visualTkImage)
        panelB.image = visualTkImage
        panelB.pack(side="right", padx=10, pady=10)
    else:
        panelA.configure(image=originalTkImage)
        panelB.configure(image=visualTkImage)
        panelA.image = originalTkImage
        panelB.image = visualTkImage


model = DefineModel()

root = Tk()
s = Label(root, text="PROGRAM DETEKSI PENYAKIT BLAS PADA TANAMAN PADI METODE DEEP LEARNING")
s.pack()
panelA = None
panelB = None
btn = Button(root, text="Select an Image", command=lambda: SelectImage(model))
btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
root.mainloop()
