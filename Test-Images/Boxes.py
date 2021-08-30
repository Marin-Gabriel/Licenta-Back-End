from cv2 import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
import numpy as np

def deskew(_img):

    grayscale = rgb2gray(_img)
    angle = determine_skew(grayscale)
    rotated = rotate(_img, angle, resize=True) * 255
    return rotated.astype(np.uint8)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#Change DPI
#original = Image.open('Test-Images/Mobile_Photos/MobPhoto_4.jpg')
original = Image.open('images/Intro.jpg')
#original = Image.open('Test-Images/WAP/1.jpeg')

original.save("DPI-Changed/300DPI.jpg", dpi=(300,300))
img = cv2.imread("DPI-Changed/300DPI.jpg")
img=deskew(img)
hImg,wImg,_=img.shape

boxes=pytesseract.image_to_data(img)
'''
for b in boxes.splitlines():
    b=b.split(' ')
    x,y,w,h=int(b[1]),int(b[2]),int(b[3]),int(b[4])
    #cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(255,0,0),1)
    cv2.putText(img,b[0],(x,hImg-y),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),3)
'''

for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b=b.split()
        print(len(b))
        if len(b)==12:
            
            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img,(x,y),(w+x,h+y),(255,0,0),1)
            cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)

plt.figure(figsize=(18,18))
plt.imshow(img)
plt.axis('off')
plt.show()

