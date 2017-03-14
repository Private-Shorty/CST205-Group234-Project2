import numpy as np
import pprint
import cv2
import PIL
from PIL import Image

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

img = cv2.imread("ABoringFellow.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#pprint.pprint(faces)

for (counter, (x,y,w,h)) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 7)
        #cv2.putText(img, "Actor #{}".format(counter+1), (x, y-10), 
        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        roi_color = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #pprint.pprint(eyes)
        for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
                        
                
basewidth = (x+w)-x
img2 = Image.open('Cowboyhat.png')
img2 = img2.resize((basewidth,int(basewidth*0.6)), PIL.Image.ANTIALIAS)
img2.save('ABetterHat.png') 



"""
picture=Image.open('ABetterHat.png')
picture=picture.convert("RGBA")

pixdata = picture.load()

width, height = picture.size
for y in range(height):
        for x in range(width):
                if pixdata[x,y] == (255, 255, 255, 255):
                        pixdata[x,y] = (255, 255, 255, 0)
picture.save("Atest.png", "PNG")
"""

#img3 = Image.open('Atest.png')
foreground = Image.open('Atest.png')
foreground.convert('RGBA')
#img4 = Image.open('ABoringFellow.png')
background = Image.open('ABoringFellow.png')



background.paste(foreground, (x, y-(h/2)), foreground)
background.save('ItWorks.png', "PNG")
#img4.paste(img3, (x,y-(h/2)))
#img4.save('ItWorks.png')
                
                
print(faces)

cv2.imwrite('image.png', img)
cv2.waitKey()