### LOOPING FOLDER FOR CHECK THE DATA CLASS ###

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import XCEPTION_MODEL
import sys
sys.path.insert(1, '\3-PostProcessing')

folder_loop = 0
file_loop = 0
filename = []
for i in np.sort(os.listdir(r"/content/drive/MyDrive/Blast_Image/Dataset_v1_80_10_10/Validation")):
    folder_loop += 1
    filename = np.append(filename, i)
    print("folder : ", folder_loop, "\n", i)
    for j in os.listdir(os.path.join(r"/content/drive/MyDrive/Blast_Image/Dataset_v1_80_10_10/Validation", i)):
        file_loop += 1
        filename = np.append(filename, j)
print("file : ", file_loop)

# Saving the data filename
np.savez(r"filename.npz", filename=filename)

# Sort data
np.sort(os.listdir(
    r"/content/drive/MyDrive/Blast_Image/Dataset_v1_80_20/Validation_20percent_6702"))

# Training the data
image_size = (700, 700)
batch_size = (16)

# Train
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    r"/content/drive/MyDrive/Blast_Image/Dataset_v1_80_20/Train_80percent_26784",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=1337,
    shuffle=True
)

# Validation
validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(
    r"/content/drive/MyDrive/Blast_Image/Dataset_v1_80_20/Validation_20percent_6702",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=1337,
    shuffle=True
)

# Test
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    r"/content/drive/MyDrive/Blast_Image/Dataset/Test",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=1337,
    shuffle=False
)

# USE XCEPTION MODEL
model = XCEPTION_MODEL.make_model(image_size + (3,), 5)

# See the model summary and hirearchy
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# Training Function
epochs = 2

callbacks = [
             tf.keras.callbacks.ModelCheckpoint(r"/content/drive/MyDrive/Blast_Weight_5_Class/training_without_validation/2_blast_saved_at_{epoch}.h5"),
            # tf.keras.callbacks.TensorBoard(log_dir = logdir, histogram_freq = 1),
             #tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
]

#model = tf.keras.models.load_model(r"/content/drive/MyDrive/Blast_Weight_5_Class/training_without_validation/blast_saved_at_16.h5")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit(
    train_generator, epochs=epochs, callbacks=callbacks
    )

df = pd.DataFrame(data=history.history)

#pertama, header=true, selanjutnya false
df.to_csv(r"/content/drive/MyDrive/Blast_Weight_5_Class/training_without_validation/Model_Recap_5_Classes.csv", mode='a', header=False, index=False)

# Evaluating the Model
model.evaluate(validation_generator)

