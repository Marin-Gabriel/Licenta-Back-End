# Import packages 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import sys
import os
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
import base64
from google_trans_new import google_translator  
from word2word import Word2word
en2fr = Word2word("en", "ro")

def verifyBox(boxW,boxH,boxX,boxY,imgH,imgW):
    #print("X1= " +str(boxX-1/2*boxW)+"Y1= " +str(boxY-1/2*boxH) +"X2= "+str(boxX+boxW*1/2)+"Y2= " +str(boxY+boxH*1/2))
    if(boxX-1/2*boxW<0 or boxY-1/2*boxH<0 or boxX+boxW*1/2>imgW or boxY+boxH*1/2>imgH):
        return False
    else:
        return True

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
      textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
      new_width = textSize[0][0]
      if (new_width <= width):
          return scale/10
    return 1

def deskew(_img):

    grayscale = rgb2gray(_img)
    angle = determine_skew(grayscale)
    #print("Angle= "+str(angle))
    
    rotated = rotate(_img, angle, resize=True) * 255
    return rotated.astype(np.uint8)

#Create MSER object
mser = cv2.MSER_create()

#Your image path i-e receipt path

img = Image.open('images/TI.jpg')

img.save("DPI-Changed/300DPI.jpg", dpi=(300,300))
img = cv2.imread("DPI-Changed/300DPI.jpg")

img=deskew(img)
deskewed=img
cv2.imwrite('images/DESS.jpg',deskewed)

#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vis = img.copy()

#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite('Threshed/text_only.jpg', text_only)
#original = cv2.imread('Threshed/text_only.jpg')
binarised=cv2.threshold(text_only, 10, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite('Threshed/bin.jpg', binarised)
img = cv2.imread('Threshed/bin.jpg', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)

img_shape=img.shape
#print(img_shape[0])
for i in range(1, nb_components):
    if(verifyBox(stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT],stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP],img_shape[0],img_shape[1])):
        cv2.rectangle(img,(int(stats[i,cv2.CC_STAT_LEFT]-1/2*stats[i,cv2.CC_STAT_WIDTH]),
        int(stats[i,cv2.CC_STAT_TOP]-1/2*stats[i,cv2.CC_STAT_HEIGHT]))
        ,(int(stats[i,cv2.CC_STAT_LEFT]+3/2*stats[i,cv2.CC_STAT_WIDTH])
        ,int(stats[i,cv2.CC_STAT_TOP]+3/2*stats[i,cv2.CC_STAT_HEIGHT])),(255,255,255),-1)
        
        #cv2.circle(img, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        #print("Hi")

    else:
        cv2.rectangle(img,(stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP]),(stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]),(255,255,255),-1)
        #(cX, cY) = centroids[i]
        #cv2.circle(img, (int(cX), int(cY)), 4, (0, 0, 255), -1)

kernel = np.ones((15, 15), 'uint8')
dilate_img = cv2.dilate(img, kernel, iterations=1)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilate_img, connectivity=4)

for i in range(1, nb_components):
    crop_img = deskewed[stats[i,cv2.CC_STAT_TOP]:stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT], stats[i,cv2.CC_STAT_LEFT]:stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH]]
    cv2.imwrite('crops/crop'+str(i)+'.jpg', crop_img)

        #(cX, cY) = centroids[i]
        #cv2.circle(dilate_img, (int(cX), int(cY)), 10, (0, 0, 255), -1)
#num_labels, labels = cv2.connectedComponents(dilate_img)
#print(num_labels)

plt.figure(figsize=(18,18))
plt.imshow(dilate_img)
plt.axis('off')
plt.show()