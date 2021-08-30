#Change DPI-OK
#Remove Shadows-OK
#Deskew-OK
#Keystone Correction-?
#Line Straightening
#Binarise-OK
#Noise Removal/Image Despeckling-OK
#Increase Sharpness and Contrast-OK 
#Dilation and Erosion

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

#Change DPI
#original = Image.open('Test-Images/Mobile_Photos/MobPhoto_4.jpg')
original = Image.open('images/Umberto.jpg')
#original = Image.open('Test-Images/WAP/1.jpeg')

original.save("DPI-Changed/300DPI.jpg", dpi=(300,300))
original = cv2.imread("DPI-Changed/300DPI.jpg")

#Remove Shadow
rgb_planes = cv2.split(original)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

#cv2.imwrite('Shadow-Removed/shadows_out_norm.jpg', result_norm)

#Deskew
rotated=deskew(result_norm)

#Despekle
dst = cv2.bilateralFilter(src=rotated, d=0, sigmaColor=100, sigmaSpace=15)
kennel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # Sharpening operation makes the image more stereo
dst1 = cv2.filter2D(dst, -1, kennel)

cv2.imwrite('Despekled/despekled.jpg', dst1)

gray=cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

cv2.imwrite('Threshed/threshed.jpg', thresh)

#Dilation
#kernel = np.ones((5,5), np.uint8)
#dilated=cv2.dilate(thresh, kernel, iterations=1)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(thresh)
text = os.linesep.join([s for s in text.splitlines() if s])
print(text)

cv2.imwrite('Pipe-Results/Result.jpg', thresh)

plt.figure(figsize=(18,18))
plt.imshow(thresh)
plt.axis('off')
plt.show()
