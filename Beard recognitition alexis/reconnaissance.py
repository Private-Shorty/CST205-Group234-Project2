#reconnaissance.py
import numpy as np
import matplotlib.pyplot as plt
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
barbe_cascade=cv2.CascadeClassifier('haarcascade_barbe.xml')

#nomfic=input("Veuillez saisir le nom de fichier de l_image a ouvrir: ")

imgcv = cv2.imread("pierre.png")
gray = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
okbarbe=0
for (x,y,w,h) in faces:    
    cv2.rectangle(imgcv,(x,y),(x+w,y+h),(255,0,0),2)
    gray1 = gray[y:y+h, x:x+w]
    color1 = imgcv[y:y+h, x:x+w]
    gray2 = gray[y+int(h/2):y+int(3*h/2), x:x+int(8*w/9)]
    color2 = imgcv[y+int(h/2):y+int(3*h/2),x:x+int(8*w/9)]
    barbes =barbe_cascade.detectMultiScale(gray2)
    for (bx,by,bw,bh) in barbes:
        cv2.rectangle(color2,(bx,by),(bx+bw,by+bh),(255,255,0),2)
        okbarbe=1     
    eyes = eye_cascade.detectMultiScale(gray1)   
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(color1,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
if(okbarbe):
       print('Barbe OUI')       
else:
       print('Barbe NON')


b,g,r = cv2.split(imgcv)
img2 = cv2.merge([r,g,b])
plt.imshow(img2)
cv2.imwrite('resultat.png', imgcv)
plt.show()


