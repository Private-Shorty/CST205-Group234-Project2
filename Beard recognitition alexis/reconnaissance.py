#reconnaissance.py
import numpy as np
import matplotlib.pyplot as plt
import cv2  # import openCv librairie 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
barbe_cascade=cv2.CascadeClassifier('haarcascade_barbe.xml') # we have creat this file for beard detection

#nomfic=input(" select the name of the file that you want open :")

imgcv = cv2.imread("pierre.png")    #open picture for test 
gray = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)#adaptation picture filter before test
faces = face_cascade.detectMultiScale(gray, 1.3, 5) #face detection
okbarbe=0       # initialise beard at 0
for (x,y,w,h) in faces:    
    cv2.rectangle(imgcv,(x,y),(x+w,y+h),(255,0,0),2) # put rectangle on face eye and beard if there is one
    gray1 = gray[y:y+h, x:x+w]
    color1 = imgcv[y:y+h, x:x+w]
    gray2 = gray[y+int(h/2):y+int(3*h/2), x:x+int(8*w/9)]
    color2 = imgcv[y+int(h/2):y+int(3*h/2),x:x+int(8*w/9)]
    barbes =barbe_cascade.detectMultiScale(gray2)
    for (bx,by,bw,bh) in barbes:
        cv2.rectangle(color2,(bx,by),(bx+bw,by+bh),(255,255,0),2)# rectangle for beard
        okbarbe=1     
    eyes = eye_cascade.detectMultiScale(gray1)   
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(color1,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)# rectangle for eye
if(okbarbe):
       print('Beard YES') # print yes if there is a beard recognition    
else:
       print('Beard NO') # print no if there is no beard 


b,g,r = cv2.split(imgcv)
img2 = cv2.merge([r,g,b])
plt.imshow(img2)
cv2.imwrite('resultat.png', imgcv) #print result image
plt.show()


