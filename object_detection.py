import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as xet
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

df = pd.read_csv('labels.csv')
df.head()
#
filename = df['filepath']

def getFilname(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('blachy',filename_image)
    return filepath_image

images_path = list(df['filepath'].apply(getFilname))
print(images_path)


#### verify image and output
# file_path = os.path.abspath(images_path[0])
# print(file_path)
# img = cv2.imread(file_path)
# cv2.imshow('example',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.rectangle(img,)

labels = df.iloc[:,1:].values
img_number = 0
image = os.path.abspath(images_path[img_number])
print(image)
img_arr = cv2.imread(image)
h,w,d = img_arr.shape
#preprocessing
load_image = load_img(image,target_size=(224,224))
load_image_arr = img_to_array(load_image)
#normalization to labels
xmin,xmax,ymin,ymax = labels[img_number]
nxmin,nxmax = xmin/w,xmax/w
nymin,nymax = ymin/h,ymax/h
label_norm = (nxmin/nxmax,nymin,nymax)

print(label_norm)
print(load_image_arr)