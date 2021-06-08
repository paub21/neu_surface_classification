import time
import pickle
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import keras
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# load dictionary (class names + class indeces)
file = open("file.dict",'rb')
object_file = pickle.load(file)
file.close()

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

#img_path = 'left_01.jpg'
#img_path = 'left_02.jpg'
#img_path = './test/pra_somdet_01.jpg'
img_path = './NEU-CLS/val2/Cr/Cr_270.jpg'
#img_path = './NEU-CLS/val2/In/In_283.jpg'
img_path = './NEU-CLS/val2/RS/RS_270.jpg'
#img_path = './test/pra_somdet_01.jpg'

# load the trained model
model = load_model('model.h5')
new_image = load_image(img_path, show=False)
start = time.time()
pred = model.predict(new_image)
end = time.time()
print(end - start)

print('Probabilities of each class:')
print(pred)
print(object_file)
print('Predicted class : %d'%pred.argmax(axis=-1))

for k in object_file:
    if object_file[k] == (pred.argmax(axis=-1)):
        print('predicted class name: %s' %k)
        #print(object_file[k])