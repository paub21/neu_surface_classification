import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import matplotlib.image as mpimg
import os

import pathlib
dataset_path = "D:/python/TransferLearning/amulet/NEU_surface_detection/NEU-CLS"

data_dir = pathlib.Path(dataset_path)
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

batch_size=30
img_height=200
img_width=200

""" PATH = "D:/python/TransferLearning/amulet/NEU_surface_detection/NEU-CLS/train/Cr/Cr_1.bmp"
image = mpimg.imread(PATH) 
plt.show()
plt.imshow(image)
plt.savefig('Crazing.png')
plt.show() """

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

train_data_gen = image_gen_train.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True, color_mode='grayscale',
                                                target_size=(img_width,img_height),
                                                class_mode='sparse'
                                                )

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(img_width,img_height),  color_mode='grayscale',
                                                 class_mode='sparse')

model = tf.keras.Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_width,img_height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(6))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 100

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
) 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()