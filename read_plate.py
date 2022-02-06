import pytesseract as pt
import predict
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
pt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def draw_bounds():
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    plt.imshow(image)
    plt.show()
    return


path = './test_images_pl/Screenshot_6.jpg'
image,coords = predict.detect_plates(path)
plt.figure(figsize=(10,8))
plt.imshow(image)
xmin,xmax,ymin,ymax = coords[0]
img = np.array(load_img(path))
roi = img[ymin-10:ymax+10,xmin-10:xmax+10]
draw_bounds()
plt.imshow(roi)
plt.show()

#extract text from img
print(pt.image_to_string(roi))
