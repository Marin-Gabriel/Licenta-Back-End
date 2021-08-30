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
import base64
from google_trans_new import google_translator  
from word2word import Word2word
en2fr = Word2word("en", "ro")

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

translator = google_translator()  



original=Image.open('images/TI.jpg')

original.save("DPI-Changed/300DPI.jpg", dpi=(300,300))
original = cv2.imread("DPI-Changed/300DPI.jpg")

deskewed=deskew(original)


#Remove Shadow
rgb_planes = cv2.split(deskewed)

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


#Despekle
dst = cv2.bilateralFilter(src=result_norm, d=0, sigmaColor=100, sigmaSpace=15)
kennel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # Sharpening operation makes the image more stereo
dst1 = cv2.filter2D(dst, -1, kennel)

#cv2.imwrite('Despekled/despekled.jpg', dst1)

gray=cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

cv2.imwrite('Threshed/threshed.jpg', thresh)

#Dilation
#kernel = np.ones((5,5), np.uint8)
#dilated=cv2.dilate(thresh, kernel, iterations=1)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

boxes=pytesseract.image_to_data(thresh)

words=""
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b=b.split()
        if len(b)==12:
            words+=b[11]+" "

translated=translate_text = translator.translate(words,lang_tgt='ro')
print(translated)

for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b=b.split()
        if len(b)==12:
            
            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(deskewed,(x,y),(w+x,h+y),(255,255,255),-1)
            #SPStripperd=''.join(e for e in b[11] if e.isalnum())
            #print(SPStripperd)
            '''
            try:
                translate_text = en2fr(SPStripperd)[0]
            except:
                translate_text=SPStripperd
            #print(translate_text +" VS "+SPStripperd)
            '''

            optimalScale=get_optimal_font_scale(b[11],w)
            cv2.putText(deskewed,b[11],(x,y+h),cv2.FONT_HERSHEY_COMPLEX,optimalScale,(0,0,0),4)

#retval,buffer=cv2.imencode('.jpg',deskewed)
#print(base64.b64encode(buffer).decode("utf-8"))

plt.figure(figsize=(18,18))
plt.imshow(deskewed)
plt.axis('off')
plt.show()
#retval,buffer=cv2.imencode('.jpg',deskewed)
#print(base64.b64encode(buffer).decode("utf-8"))