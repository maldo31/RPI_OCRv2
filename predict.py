import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

# load model
model = tf.keras.models.load_model('./model/object_detection_pl_v6.h5')
print('model loaded sucessfully')
path = 'test_images_pl/Screenshot_9.jpg'

#create pipeline
def detect_plates(path):
    # read image
    image = load_img(path)
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    # prepare data
    image_arr_224 = img_to_array(image1)/255.0
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    # predict plates
    coords = model.predict(test_arr)
    # denormalzie image
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounds around plate
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    return image, coords


