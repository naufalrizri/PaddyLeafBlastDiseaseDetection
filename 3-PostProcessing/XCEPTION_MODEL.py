# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:00:26 2021

@author: Naufal Rizki
"""


"""XCEPTION MODEL"""

from keras import layers
import keras

### Augmentation
data_augmentation = keras.Sequential(
    [
     layers.experimental.preprocessing.RandomTranslation(.1, .1, fill_mode='reflect', interpolation='bilinear')
     ]
    )

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    
    x = layers.experimental.preprocessing.Rescaling(1.0/255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    previous_block_activation = x
    
    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        
        # Residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x
        
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation="sigmoid"
        units=1
    else:
        activation="softmax"
        units=num_classes
    
    x = layers.Dropout(.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)