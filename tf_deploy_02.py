import time
import pickle
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
#import keras
#import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

# load dictionary (class names + class indeces)
file = open("file.dict",'rb')
object_file = pickle.load(file)
file.close()

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input) #included in our dependencies
test_generator = test_datagen.flow_from_directory('./NEU-CLS/val/', # this is where you specify the path to the main data folder
                                                 target_size = (224,224),
                                                 color_mode = 'rgb',
                                                 batch_size = 8,
                                                 class_mode = 'categorical',
                                                 shuffle = False,
                                                 seed=42)

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
  
# test images   

#img_path = './NEU-CLS/val2/1.jpg'
#img_path = 'left_02.jpg'
#img_path = './test/pra_somdet_01.jpg'
#img_path = './test/03/01.jpg'
#img_path = './test/pra_somdet_01.jpg'


# load the trained model
model = load_model('model.h5')

pred = model.predict_generator(test_generator) 
pred_class_id=np.argmax(pred,axis=1)
labels=(test_generator.class_indices)
labels2=dict((v,k) for k,v in labels.items())
predictions=[labels2[k] for k in pred_class_id]


print(pred_class_id)
#print(labels)
#print(labels2)
#print(predictions)
