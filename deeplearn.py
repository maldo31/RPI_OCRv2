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
model = tf.keras.models.load_model('./model/object_detection_pl_v9.h5')


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
    h, w, d = img.shape
    cords = detect_plates(path, filename)
    print(cords)
    xmin, xmax, ymin, ymax = cords[0]
    #change color to shades
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #little blur so pytesseract works better
    img = cv2.medianBlur(img, 3)

    #### 4 different thresh
    text = ''
    for px in range(5, 120, 5):
        if not text or text.isspace() or len(text) < 7 or len(text) > 8 or not text[0] in first_letters:
            roi0 = img[ymin - px:ymax + px, xmin - px-2:xmax + px+2]
            text = print_if_ok(roi0, "no thresh")
        else:
            roi_bgr = cv2.cvtColor(roi0, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(ROI_PATH, filename), roi_bgr)
            return text
    lol, thresh_temp = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    for px in range(5, 120, 5):
        if not text or text.isspace() or len(text) < 7 or len(text) > 8 or not text[0] in first_letters:
            thresh1 = thresh_temp[ymin - px:ymax + px-2, xmin - px:xmax + px+2]
            text = print_if_ok(thresh1, "thresh1")
        else:
            roi_bgr = cv2.cvtColor(thresh1, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(ROI_PATH, filename), roi_bgr)
            return text
    lol, thresh_temp = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
    for px in range(5, 120, 5):
        if not text or text.isspace() or len(text) < 7 or len(text) > 8 or not text[0] in first_letters:
            thresh2 = thresh_temp[ymin - px:ymax + px, xmin - px-2:xmax + px+2]
            text = print_if_ok(thresh2, "thresh2")
            # thresh4 = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            # print_if_ok(thresh4,"thresh4")
        else:
            roi_bgr = cv2.cvtColor(thresh2, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(ROI_PATH, filename), roi_bgr)
            return text
    lol, thresh_temp = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
    for px in range(5, 120, 5):
        if not text or text.isspace() or len(text) < 7 or len(text) > 8 or not text[0] in first_letters:
            print(px)
            thresh3 = thresh_temp[ymin - px:ymax + px, xmin - px-2:xmax + px+2]
            text = print_if_ok(thresh3, "thresh3")
        else:
            roi_bgr = cv2.cvtColor(thresh3, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(ROI_PATH, filename), roi_bgr)
            return text

    img = img[ymin - px:ymax + px, xmin - px:xmax + px]
    roi_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(ROI_PATH, filename), roi_bgr)
    print(text)
    return text


def text_reck(path,filename):




    img = np.array(load_img(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    lol, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

    text = pt.image_to_string(img)
    roi_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(ROI_PATH, filename), roi_bgr)
    print(text)

    return text


def print_if_ok(img_to_read, thresh):
    OCRtext = pt.image_to_string(img_to_read)
    OCRtext, thrash = polish_plates_check(OCRtext)
    if len(OCRtext) >= 7 and len(OCRtext) <= 8 and OCRtext[0] in first_letters:
        print(thresh + ": " + OCRtext)
        print("\nbefore cleaning " + thrash + "\n\n")
    return OCRtext


def polish_plates_check(plt_str):
    unclean = plt_str
    plt_str = plt_str.replace('|', 'I')
    plt_str = re.sub('[^A-Z0-9]+', '', plt_str)
    return plt_str, unclean
