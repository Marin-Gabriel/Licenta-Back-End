import numpy as np
import cv2
import textwrap 

img =cv2.imread("DPI-Changed/300DPI.jpg")

font = cv2.FONT_HERSHEY_SIMPLEX

text = "type: car, color: white, number: 123456"
#to automatically wrap text => wrapped_text = textwrap.wrap(text, width=10)
wrapped_text = ['Type: car','Color: white','Number: 123456']
x, y = 10, 40
font_size = 1
font_thickness = 2
H=500
W=3500
cv2.rectangle(img,(0,0),(W,H),(255,255,255),-1)



indexMax=0
index=0
max=0
for line in wrapped_text:
    if(len(line)>max):
        max=len(line)
        indexMax=index
    index+=1

textsize = cv2.getTextSize(wrapped_text[indexMax], font, font_size, font_thickness)[0]
while(textsize[0]<W*0.8):
    font_size+=0.1
    textsize = cv2.getTextSize(wrapped_text[indexMax], font, font_size, font_thickness)[0]
print(textsize)

i = 0
for line in wrapped_text:
    
    #textsize[0]=lungime in pixeli
    #textsize[1]=inaltime in pixeli
    #textsize = (W*0.8,H*0.25)

    gap = int(textsize[1]/2)

    y = int((y+ textsize[1])) + gap
    x = 0#for center alignment => int((img.shape[1] - textsize[0]) / 2)
    print(y)
    print(textsize[1])
    cv2.putText(img, line, (x, y), font,
                font_size, 
                (0,0,0), 
                font_thickness, 
                lineType = cv2.LINE_AA)
    i +=1

cv2.imwrite('images/textt.jpg',img)