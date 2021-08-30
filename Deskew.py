from PIL import Image, ImageEnhance, ImageFilter
from cv2 import cv2
import pytesseract
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew

def deskew(_img):

    grayscale = rgb2gray(_img)
    angle = determine_skew(grayscale)
    rotated = rotate(_img, angle, resize=True) * 255
    return rotated.astype(np.uint8)

original=cv2.imread('images/Henry.jpg')
rotated=deskew(original)

#h, w, c = original.shape
#original=cv2.resize(original, (int(w/2),int(h/2)))

gray=cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(thresh)
text = os.linesep.join([s for s in text.splitlines() if s])
print(text)

plt.figure(figsize=(18,18))
plt.imshow(thresh)
plt.axis('off')
plt.show()
