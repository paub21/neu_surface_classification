# add some comments
import pickle
import pandas as pd
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import keras
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from keras.optimizers import Adam
# load and evaluate a saved model
from keras.models import load_model

batch_size = 30
num_epochs_fast = 10
num_epochs = 100
frozen_layers = 20

#train_datagen = ImageDataGenerator(rotation_range=90) #included in our dependencies
train_datagen = ImageDataGenerator(
                        preprocessing_function = preprocess_input, 
                        rotation_range=90, 
                        zoom_range=0.15,
	                    width_shift_range=0.2,
	                    height_shift_range=0.2,
	                    shear_range=0.15,
	                    horizontal_flip=True,
	                    fill_mode="nearest") #included in our dependencies

""" train_datagen = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest") """
train_generator = train_datagen.flow_from_directory('./NEU-CLS/train/', # this is where you specify the path to the main data folder
                                                 target_size = (224,224),
                                                 color_mode = 'rgb',
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical',
                                                 shuffle = True,
                                                 seed=42)

#test_datagen = ImageDataGenerator() #included in our dependencies
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input) #included in our dependencies

test_generator = test_datagen.flow_from_directory('./NEU-CLS/val/', # this is where you specify the path to the main data folder
                                                 target_size = (224,224),
                                                 color_mode = 'rgb',
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical',
                                                 shuffle = True,
                                                 seed=42)

# ตรวจนับจำนวน classes ทั้งหมด โดยนับจากจำนวน folders ที่เจอ
num_of_classes = len(train_generator.class_indices)
print('number of classes : %d ' %num_of_classes) 

#imports the mobilenet model and discards the last 1000 neuron layer.
base_model = MobileNet(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)) 

# mobilenet มี 87 layer (+1 output layer ที่ถูกเอาออกไป)
cnt = 0
for layer in base_model.layers[:]:
    cnt = cnt + 1
print('number of layers of base_model : %d' %cnt) 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation = 'relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(512, activation = 'relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(512, activation = 'relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(512, activation = 'relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.
#x = Dense(1024, activation = 'relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.
#x = Dense(1024, activation = 'relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.

preds = Dense(num_of_classes, activation = 'softmax')(x) #final layer with softmax activation

model = Model(inputs = base_model.input, outputs = preds)

print(model.summary())

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# train the model on the new data for a few epochs
step_size_train = train_generator.n//train_generator.batch_size
model.fit_generator(generator = train_generator,
                    steps_per_epoch = step_size_train,
                    epochs = num_epochs_fast,
                    verbose=1)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:frozen_layers]:
    layer.trainable = False
for layer in model.layers[frozen_layers:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=num_epochs,
                    verbose=1)

print('End of the training session')

_, acc = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)

print('> %.3f' % (acc * 100.0))

# save the model to disk
model.save("model.h5")

print('Model has been saved')

filehandler = open("file.dict","wb")
pickle.dump(train_generator.class_indices,filehandler)
filehandler.close()