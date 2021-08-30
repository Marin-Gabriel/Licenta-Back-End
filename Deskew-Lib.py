import numpy as np
from cv2 import cv2
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from matplotlib import pyplot as plt

def deskew(_img):

    grayscale = rgb2gray(_img)
    angle = determine_skew(grayscale)
    rotated = rotate(_img, angle, resize=True) * 255
    return rotated.astype(np.uint8)

original=cv2.imread('Shadow-Removed/shadows_out_norm.jpg')
rotated=deskew(original)
plt.figure(figsize=(18,18))
plt.imshow(rotated)
plt.axis('off')
plt.show()

