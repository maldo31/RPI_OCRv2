import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as xet
from tensorflow.keras.applications import MobileNetV2,InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

df = pd.read_csv('labels.csv')
df.head()
#
filename = df['filepath']

def getFilname(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('test_images_pl',filename_image)
    return filepath_image

images_path = list(df['filepath'].apply(getFilname))
# print(images_path)


#### verify image and output
# file_path = os.path.abspath(images_path[0])
# print(file_path)
# img = cv2.imread(file_path)
# cv2.imshow('example',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.rectangle(img,)

labels = df.iloc[:,1:].values

data = []
output = []
for img_number in range(len(images_path)):
    image = os.path.abspath(images_path[img_number])
    # print(image)
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
    #preprocessing
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 #normalizations
    #normalization to labels
    xmin,xmax,ymin,ymax = labels[img_number]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) #normalized output
    # append to data and output
    data.append(norm_load_image_arr)
    output.append(label_norm)

x = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)
# print(x.shape,y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False,
                                     input_tensor=Input(shape=(224,224,3)))
inception_resnet.trainable=False
#############
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
#### 3 layers
headmodel = Dense(1000,activation="relu")(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(100,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)
###### model
model = Model(inputs=inception_resnet.input,outputs=headmodel)

##### compile model
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
model.summary()

##### train model
tfb =TensorBoard('object_detection_v6')
history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=250
                    ,validation_data=(x_test,y_test),
                    callbacks=[tfb])

model.save('./model/object_detection_pl_v6.h5')