import cv2
from random import randrange
#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #classifier-detector

#choose an image to detect faces in
img =cv2.imread('raghul.jpg')

#must convert to grayscale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#to detect faces
face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
#detectMultiScale will datect objects of different sizes in the input image.

# [[220 162 162 162]]-output-where the face is present

#draw rectangles around the faces
for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),4)#128 to 256 bright colours
    #cv2.rectangle(img,((220,162),(162,162)),greencolor(b,g,r),thickness of rectangle)
#to display image
cv2.imshow('_face_detection',img)

#waitkey is to pause the image      # 0 indicates infinity secondsfa
cv2.waitKey(0)
img.release()
cv2.destroyAllWindows()