#project2, can ID beards and add hats!
import numpy as np
import matplotlib.pyplot as plt
import cv2 # import openCv librairie
import pprint
import PIL
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
barbe_cascade=cv2.CascadeClassifier('haarcascade_barbe.xml') # we have created this file for beard detection

picture = raw_input("Please enter the name of the picture you would like edited: ") #Allows user to enter the image
answer = input("Enter 1 for beard identification, and 2 to give them a nice hat: ") #Allows user to select function

if (answer == 1):
    imgcv = cv2.imread(picture)    #open picture
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
    #plt.imshow(img2)
    cv2.imwrite('FinalImage.png', imgcv) #print result image
    plt.show()
elif (answer == 2):
    imgcv = cv2.imread(picture)
    gray = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #pprint.pprint(faces)

    for (counter, (x,y,w,h)) in enumerate(faces):
        cv2.rectangle(imgcv, (x, y), (x+w, y+h), (255, 0, 0), 7)
        #cv2.putText(img, "Actor #{}".format(counter+1), (x, y-10), 
        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        roi_color = imgcv[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #pprint.pprint(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
                        
                
        basewidth = (x+w)-x
        img2 = Image.open('Cowboyhat.png')
        img2 = img2.resize((basewidth,int(basewidth*0.6)), PIL.Image.ANTIALIAS)
        img2.save('ABetterHat.png') 

    #img3 = Image.open('Atest.png')
        foreground = Image.open('Atest.png')
        foreground.convert('RGBA')
    #img4 = Image.open('ABoringFellow.png')
        background = Image.open(picture)



        background.paste(foreground, (x, y-(h/2)), foreground)
        background.save('FinalImage.png', "PNG")
    
        cv2.imwrite('Image.png', imgcv)
        cv2.waitKey()
else:
    print("That is not a valid response, exiting application...")
