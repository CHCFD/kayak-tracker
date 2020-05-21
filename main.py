# -*- coding: utf-8 -*-
"""
main script
Chan Chi Hin
"""

import cv2 as cv
import numpy as np

# return resized with fixed interpolation

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

# Concatenate images of different widths vertically
def vconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation) for im in im_list]
    return cv.vconcat(im_list_resize)

# Loading Video
cap = cv.VideoCapture('./videos/test.avi')

while cap.isOpened():
    
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    # Finding Boat
    diff = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    ret,diff = cv.threshold(diff,210,255,0)
    contours, hier = cv.findContours(diff,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    
    # Retrieving Rectangle
    arr = []
    for cnt in contours:
        (x, y, w, h) = cv.boundingRect(cnt)
        # Defining Area of contour limit
        if 400 < cv.contourArea(cnt) < 50000:
            if y > 500:
                # superposition, colour, thickness
                arr.append((x,y))
                arr.append((x+w,y+h))
                
    # Draw BoundingBox
    x,y,w,h = cv.boundingRect(np.asarray(arr))
    cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),3)
    cv.putText(frame1, 'KAYAK', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 2.5, (36,255,12), 5)
    
    
    # Finding LifeJacket
    lifejacket = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)
    # lower mask (0-10)
    mask1 = cv.inRange(lifejacket, (0,50,20), (5,255,255))
    mask2 = cv.inRange(lifejacket, (175,50,20), (180,255,255))
    mask = cv.bitwise_or(mask1, mask2 )
    croped = cv.findContours(mask, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
    
    if len(croped)>0:
        blue_area = max(croped, key=cv.contourArea)
        (xg,yg,wg,hg) = cv.boundingRect(blue_area)
        cv.rectangle(frame1,(xg,yg),(xg+wg, yg+hg),(0,0,255),2)
        cv.putText(frame1, 'LIFEJACKET', (xg, yg-10), cv.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255),  5) 
        
    # Resizing image
    frame1 = ResizeWithAspectRatio(frame1, width = 1280)
     
    cv.imshow('IMG', frame1)


    if cv.waitKey(40)==27:
        break

cv.destroyAllWindows()
cap.release()
