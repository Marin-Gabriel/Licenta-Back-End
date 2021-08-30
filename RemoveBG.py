import cv2 
import numpy as np 

bgray = cv2.imread('Test-Images/Mobile_Photos/MobPhoto_4.jpg')[...,0]

blured1 = cv2.medianBlur(bgray,3)
blured2 = cv2.medianBlur(bgray,51)
divided = np.ma.divide(blured1, blured2).data
normed = np.uint8(255*divided/divided.max())
th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)

dst = np.vstack((bgray, blured1, blured2, normed, threshed)) 
kernel = np.ones((3,3), np.uint8)
dilated=cv2.erode(dst, kernel, iterations=1)
cv2.imwrite("dst.jpg", dilated)
