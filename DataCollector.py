from cv2 import cv2
import os
import numpy as np


cap = cv2.VideoCapture(0)

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


img_counter = 0
while(True):
    got_face = False
    got_eyes = False
    ret, frame = cap.read()
    # on each frame apply face and eyes haarcascade
    if ret:

        if img_counter % 5 == 0:
            dest = "./data/test"
        else:
            dest = "./data/train"
            
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            if len(eyes)==2:
                got_eyes = True
                eye_frame = roi_color
                cv2.imshow("frame", eye_frame)
            else:
                got_eyes = False

            k = cv2.waitKey(1)
            # escape key
            # if k%256 == 27:
            #     break

            # 1 - 9 
            if k%256 == 49 and got_eyes:
                print("captured 1")
                img_name = f"{dest}/1/opencv_frame{img_counter}.png"
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1

            if k%256 == 50 and got_eyes:
                print("captured 2")
                img_name = f"{dest}/2/opencv_frame{img_counter}.png"
                print(img_name)
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1


            if k%256 == 51 and got_eyes:
                print("captured 3")
                img_name = f"{dest}/3/opencv_frame{img_counter}.png"
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1


            if k%256 == 52 and got_eyes:
                print("captured 4")
                img_name = f"{dest}/4/opencv_frame{img_counter}.png"
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1


            if k%256 == 53 and got_eyes:
                print("captured 5")
                img_name = f"{dest}/5/opencv_frame{img_counter}.png"
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1

            if k%256 == 54 and got_eyes:
                print("captured 6")
                img_name = f"{dest}/6/opencv_frame{img_counter}.png"
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1

            if k%256 == 55 and got_eyes:
                print("captured 7")
                img_name = f"{dest}/7/opencv_frame{img_counter}.png"
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1

            if k%256 == 56 and got_eyes:
                print("captured 8")
                img_name = f"{dest}/8/opencv_frame{img_counter}.png"
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1

            if k%256 == 57 and got_eyes:
                print("captured 9")
                img_name = f"{dest}/9/opencv_frame{img_counter}.png"
                cv2.imwrite(img_name, eye_frame)
                img_counter +=1

    k = cv2.waitKey(1)
    # escape key
    if k%256 == 27:
        break



cap.release()
cv2.destroyAllWindows()