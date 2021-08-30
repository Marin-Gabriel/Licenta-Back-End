import cv2
import numpy as np

img = cv2.imread('Threshed/bin.jpg', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
num_labels, labels_im = cv2.connectedComponents(img)


print(num_labels)

