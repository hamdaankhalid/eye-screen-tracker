from cv2 import cv2
import os
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow import keras
from graphics import *
import operator


win = GraphWin("Heat Mapp", 1250, 700)


def setup():
    # for each rectangle in the window, create one 9*9 matrix
    p1 = Point(0,0)
    p2 = Point(416, 233)

    p3 = Point(416, 0)
    p4 = Point(832, 233)

    p5 = Point(832, 0)
    p6 = Point(1250, 233)

    # *************

    p7 = Point(0, 233)
    p8 = Point(416, 466)

    p9 = Point(416,233)
    p10 = Point(832, 466)

    p11 = Point(832, 233)
    p12 = Point(1250, 466)

    # *************


    p13 = Point(0, 466)
    p14 = Point(416, 700)

    p15 = Point(416, 466)
    p16 = Point(832, 700)

    p17 = Point(832, 466)
    p18 = Point(1250,7200)

    # *************


    r_1 = Rectangle(p1,p2)
    r_2 = Rectangle(p3,p4)
    r_3 = Rectangle(p5,p6)
    r_4 = Rectangle(p7,p8)
    r_5 = Rectangle(p9,p10)
    r_6 = Rectangle(p11,p12)
    r_7 = Rectangle(p13,p14)
    r_8 = Rectangle(p15,p16)
    r_9 = Rectangle(p17,p18)
    
    tiles = [r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9]
    rects = {}
    for j in range(9):
        j = j+1
        name_1 = f"r_{j}"
        rects[name_1] = tiles[j-1]
        tiles[j-1].draw(win)
    return rects
    

def predict_frame(frame_array):
    img_array = np.array([frame_array])
    img = preprocess_input(img_array)
    pred = my_eye_model.predict(img)[0]
    # find index+1 of max value in pred if greater than .5 else return none
    if max(pred)>=0.7:
        ind = np.argmax(pred) + 1
        return ind
    else:
        return None


cap = cv2.VideoCapture(0)

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

my_eye_model = keras.models.load_model("eye_screen_model")

rects = setup()

img_counter = 0
while(True):
    got_face = False
    got_eyes = False
    ret, frame = cap.read()
    # on each frame apply face and eyes haarcascade
    if ret: 
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes)==2:
                got_eyes = True
                eye_frame = roi_color

                # process eye_frame to print predictions fast
                preds = predict_frame(eye_frame)
                if preds:
                    rect_1_key= f"r_{preds}"
                    
                    rect_1 = rects[rect_1_key]

                    rect_1.setFill('red')
                    
                    cv2.imshow("frame", eye_frame)

                    rect_1.setFill('white')
                else:
                    cv2.imshow("frame", eye_frame)
           
            else:
                got_eyes = False

            

    k = cv2.waitKey(1)
    # escape key
    if k%256 == 27:
        break



cap.release()
cv2.destroyAllWindows()
win.close()
