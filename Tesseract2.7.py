# Import packages 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import textwrap 
from PIL import Image, ImageEnhance, ImageFilter,ImageFont
import pytesseract
import sys
import os
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
import base64
import unidecode
#from google_trans_new import google_translator  
from googletrans import Translator
from word2word import Word2word
en2fr = Word2word("en", "ro")
#translator = google_translator()  
translator = Translator()


def verifyBox(boxW,boxH,boxX,boxY,imgH,imgW):
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
    rotated = rotate(_img, angle, resize=True) * 255
    return rotated.astype(np.uint8)

def determineMaxLengthIndex(List):
    i=0
    indexMax=0
    maxLength=0
    for Line in List:
        if(len(Line)>maxLength):
            indexMax=i
            maxLength=len(Line)
        i+=1
    return indexMax

def extractText(_img):
    rgb_planes = cv2.split(_img)

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

    #Despekle
    dst = cv2.bilateralFilter(src=result_norm, d=0, sigmaColor=100, sigmaSpace=15)
    kennel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # Sharpening operation makes the image more stereo
    dst1 = cv2.filter2D(dst, -1, kennel)

    gray=cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    #cv2.imwrite('Threshed/threshed.jpg', thresh)

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_data(thresh)

    words=""
    lineCount=0

    blockNumber=0
    linesPerBlock=0
    #print(text)

    for x,b in enumerate(text.splitlines()):
        if x!=0:
            b=b.split()
            if(b[2]!=blockNumber):
                lineCount+=int(linesPerBlock)
                blockNumber=b[2]
                linesPerBlock=0
            if len(b)==12:
                words+=b[11]+" "
                if(int(b[4])>=int(linesPerBlock)):
                    linesPerBlock=b[4]
    lineCount+=int(linesPerBlock)

    if(words==""):
        return(False,0,"")
    else:
        return (True,lineCount,words)

#------------------------------------------------------------------------------------------------------------------------------------
#Create MSER object
mser = cv2.MSER_create()

#Your image path i-e receipt path

img = Image.open('images/'+sys.argv[1])

img.save("DPI-Changed/300DPI.jpg", dpi=(300,300))
img = cv2.imread("DPI-Changed/300DPI.jpg")

img=deskew(img)
deskewed=img
#cv2.imwrite('images/DESS.jpg',deskewed)

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

text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite('Threshed/text_only.jpg', text_only)
binarised=cv2.threshold(text_only, 10, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite('Threshed/bin.jpg', binarised)
img = cv2.imread('Threshed/bin.jpg', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)

img_shape=img.shape
for i in range(1, nb_components):
    if(verifyBox(stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT],stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP],img_shape[0],img_shape[1])):
        cv2.rectangle(img,(int(stats[i,cv2.CC_STAT_LEFT]-1/2*stats[i,cv2.CC_STAT_WIDTH]),
        int(stats[i,cv2.CC_STAT_TOP]-1/2*stats[i,cv2.CC_STAT_HEIGHT]))
        ,(int(stats[i,cv2.CC_STAT_LEFT]+3/2*stats[i,cv2.CC_STAT_WIDTH])
        ,int(stats[i,cv2.CC_STAT_TOP]+3/2*stats[i,cv2.CC_STAT_HEIGHT])),(255,255,255),-1)
    else:
        cv2.rectangle(img,(stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP]),(stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]),(255,255,255),-1)

cv2.imwrite('images/515.jpg',img)

#kernel = np.ones((15, 15), 'uint8')
#dilate_img = cv2.dilate(img, kernel, iterations=1)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
'''
result=cv2.medianBlur(deskewed,99)
M=np.ones(deskewed.shape,dtype="uint8")*100
result=cv2.subtract(result,M)
'''
final_result=deskewed

for i in range(1, nb_components):
    crop_img = deskewed[stats[i,cv2.CC_STAT_TOP]:stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT], stats[i,cv2.CC_STAT_LEFT]:stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH]]
    #cv2.imwrite('crops/crop'+str(i)+'.jpg', crop_img)
    (foundText,lineCount,extractedText)=extractText(crop_img)
    #print(str(i)+" "+str(lineCount))
    if(foundText):
        #print('Nr linii gasite '+str(lineCount))
        #f = open("foundText.txt","a")
        #f.write(extractedText+'\n')
        #f.close()
        try:
            translatedText=translator.translate(extractedText,dest=sys.argv[2])
            translatedText=translatedText.text
            #translatedText.strip('?')
        except:
            #print("FAIL")
            translatedText=''
        #cv2.rectangle(deskewed,(stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP]),(stats[i,cv2.CC_STAT_WIDTH]+stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_HEIGHT]+stats[i,cv2.CC_STAT_TOP]),(255,255,255),-1)
        
        try:
            ROI=deskewed[stats[i,cv2.CC_STAT_TOP]:stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT],stats[i,cv2.CC_STAT_LEFT]:stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH]]
            #for arr in ROI:
                #print(arr[0])
            blur = cv2.medianBlur(ROI, 99)
            M=np.ones(ROI.shape,dtype="uint8")*100
            blur=cv2.subtract(blur,M)
            final_result[stats[i,cv2.CC_STAT_TOP]:stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT],stats[i,cv2.CC_STAT_LEFT]:stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH]]=blur
        except:
            #print('Blur fail')
            pass
        

        lettersPerLine=int(len(translatedText)/int(lineCount))
        wrapped_text = textwrap.wrap(translatedText, width=lettersPerLine)
        #x, y = stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP]

        indexMax=determineMaxLengthIndex(wrapped_text)
        #print(wrapped_text[indexMax])
        font_size=0.01
        font_thickness = 1
        #print('Nr linii desenate '+str(len(wrapped_text)))
        if(len(wrapped_text)>=1):
            textsize = cv2.getTextSize(wrapped_text[indexMax], cv2.FONT_HERSHEY_COMPLEX, font_size, font_thickness)[0]


            while(textsize[0]>stats[i,cv2.CC_STAT_WIDTH]*0.7):
                font_size-=0.01
                #print(font_size)
                textsize = cv2.getTextSize(wrapped_text[indexMax], cv2.FONT_HERSHEY_COMPLEX, font_size, font_thickness)[0]

            while(textsize[0]<stats[i,cv2.CC_STAT_WIDTH]*0.7):
                font_size+=0.01
                textsize = cv2.getTextSize(wrapped_text[indexMax], cv2.FONT_HERSHEY_COMPLEX, font_size, font_thickness)[0]

            gap = int(textsize[1]*0.5)
            #print(str(textsize[0])+" VS "+str(stats[i,cv2.CC_STAT_WIDTH]*0.8))
            y=0
            for j, line in enumerate(wrapped_text):

                y = int(stats[i,cv2.CC_STAT_TOP] +(gap+textsize[1])*(j+1))+gap
                x = int(stats[i,cv2.CC_STAT_LEFT]*1.15 )

                cv2.putText(final_result,
                unidecode.unidecode(line),
                (x, y),
                cv2.FONT_HERSHEY_COMPLEX,
                font_size, 
                (255,255,255), 
                font_thickness, 
                lineType = cv2.LINE_AA)
        
retval,buffer=cv2.imencode('.jpg',final_result)
print(base64.b64encode(buffer).decode("utf-8"))

#cv2.putText(deskewed,translatedText,(stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)
'''
cv2.imwrite('images/RESULT.jpg',final_result)

plt.figure(figsize=(18,18))
plt.imshow(final_result)
plt.axis('off')
plt.show()
'''