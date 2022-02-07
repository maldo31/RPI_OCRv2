import pytesseract as pt
import predict
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import re

BASE_PATH = os.getcwd()
PREDICT_PATH = os.path.join(BASE_PATH, 'static/predict/')
ROI_PATH = os.path.join(BASE_PATH, 'static/roi/')
first_letters = "BCDEFGKLNOPRSTWZ"

pt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
model = tf.keras.models.load_model('./model/object_detection_pl_v6.h5')


def detect_plates(path, filename):
    # read image
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    image1 = load_img(path, target_size=(224, 224))
    # prepare data
    image_arr_224 = img_to_array(image1) / 255.0
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    # predict plates
    coords = model.predict(test_arr)
    # denormalzie image
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounds around plate
    xmin, xmax, ymin, ymax = coords[0]
    xmin -= 5
    ymin -= 5
    xmax += 5
    xmax += 5
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(PREDICT_PATH)
    print(filename)
    cv2.imwrite(os.path.join(PREDICT_PATH, filename), image_bgr)
    return coords


def OCR(path, filename):
    img = np.array(load_img(path))
    cords = detect_plates(path, filename)
    print(cords)
    xmin, xmax, ymin, ymax = cords[0]
    roi = img[ymin:ymax, xmin:xmax]
    text = pt.image_to_string(roi)
    text = polish_plates_check(text)
    for px in range(5, 120, 5):
        if not text or text.isspace() or len(text) < 7 or len(text) > 8 or not text[0] in first_letters:
            roi = img[ymin - px:ymax + px, xmin - px:xmax + px]
            # can experiment with some threshold
            # lol, thresh1 = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
            text = pt.image_to_string(roi)
            text = polish_plates_check(text)
            print(px)
            print(text)
        else:
            break
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(ROI_PATH, filename), roi_bgr)
    print(text)
    return text

def text_reck(path):
    img = np.array(load_img(path))
    lol, img= cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    text = pt.image_to_string(img)

    text = polish_plates_check(text)
    return text


def polish_plates_check(plt_str):
    plt_str = re.sub('[^A-Z0-9]+', '', plt_str)
    return plt_str
