'''
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
im = Image.open("2.jpg") # the second one 
im = im.filter(ImageFilter.MedianFilter())
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(1)
im = im.convert('1')
im.save('temp2.jpg')
text = pytesseract.image_to_string(Image.open('temp2.jpg'))
print(text)


from PIL import Image, ImageEnhance, ImageFilter
from cv2 import cv2
import pytesseract
import sys

original=cv2.imread('images/'+sys.argv[1])

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

scale_percent = 200

#calculate the 50 percent of original dimensions
width = int(original.shape[1] * scale_percent / 100)
height = int(original.shape[0] * scale_percent / 100)
dsize = (width, height)

# resize image
rescaled = cv2.resize(original, dsize)

blured = cv2.bilateralFilter(rescaled,9,75,75)

img_grey = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)

tresholded=cv2.adaptiveThreshold(img_grey ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#cv2.imshow("window",tresholded)

text = pytesseract.image_to_string(tresholded)
print(text)

#cv2.waitKey(0)

'''

from PIL import Image, ImageEnhance, ImageFilter
from cv2 import cv2
import pytesseract
import sys
import os
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
import numpy as np

def deskew(_img):

    grayscale = rgb2gray(_img)
    angle = determine_skew(grayscale)
    rotated = rotate(_img, angle, resize=True) * 255
    return rotated.astype(np.uint8)
original=cv2.imread('images/'+sys.argv[1])

#original.save("DPI-Changed/300DPI.jpg", dpi=(300,300))
#original = cv2.imread("DPI-Changed/300DPI.jpg")

original=deskew(original)

h, w, c = original.shape
original=cv2.resize(original, (int(w/2),int(h/2)))
gray=cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(thresh)
text = os.linesep.join([s for s in text.splitlines() if s])
print(text)
'''
thresh=cv2.resize(thresh,(int(w/8),int(h/8)))
cv2.imshow('threshold', thresh)
cv2.waitKey(0)
'''