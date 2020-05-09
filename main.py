# -*- coding: utf-8 -*-
"""
main script
Chan Chi Hin
"""

import cv2 as cv

cap = cv.VideoCapture('./videos/test.avi')


while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    diff = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    ret,diff = cv.threshold(diff,210,255,0)


    contours, hier = cv.findContours(diff,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    
    # Drawing Rectangle
    for cnt in contours:
        (x, y, w, h) = cv.boundingRect(cnt)
        # Defining Area of contour limit
        if 500 < cv.contourArea(cnt) < 50000:
            if y > 500:
                # superposition, colour, thickness
                cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),3)
                
    cv.imshow('IMG', frame1)
    if cv.waitKey(100) == 7:
        break

cv.destroyAllWindows()
cap.release()
