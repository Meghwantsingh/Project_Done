import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10Epochs.h5')

image=cv2.imread('C:\\YMegwork\\AMegwork\\Brain MRI _ML_2022_project\\Brain Tumor Project_2022\\Brain MRI Tumor project\\pred\\pred0.jpg')

img = Image.fromarray(image)

img = img.resize((64,64))

img =np.array(img)

input_img=np.expand_dims(img, axis=0)

result= model.predict(input_img)

print(result)
if(result == 0):
    print("NOT have Tumor [0]\n")
else:
    print(" Tumor Detected [1]\n")
    

image=cv2.imread('C:\\YMegwork\\AMegwork\\Brain MRI _ML_2022_project\\Brain Tumor Project_2022\\Brain MRI Tumor project\\pred\\pred45.jpg')

img = Image.fromarray(image)

img = img.resize((64,64))

img =np.array(img)

input_img=np.expand_dims(img, axis=0)

result= model.predict(input_img)

print(result)
if(result == 0):
    print("NOT have Tumor [0]\n")
else:
    print(" Tumor Detected [1]\n")
