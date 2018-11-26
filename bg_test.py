import numpy as np
import cv2 as cv
import boundinrect as bdr

cap = cv.VideoCapture('videos/square.mp4')
_,frame = cap.read()
frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

B = np.zeros((frame.shape[0], frame.shape[1]))
B_norm = np.zeros((frame.shape[0], frame.shape[1]))

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,3))
kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

while(1):
    frame2 = frame
    _,frame_rgb = cap.read()
    frame = cv.cvtColor(frame_rgb, cv.COLOR_BGR2GRAY)

    Di = cv.absdiff(frame, frame2)
    _,Di_T = cv.threshold(Di, 20, 255, cv.THRESH_BINARY)
   #Di_T = float(Di_T)


    Di_T_F = 0.01*Di_T
    frame_F = Di_T_F*frame2
    #print(frame_F)
    #print('----------------------------------------------------')

    Di_T_B = 0.1*(1 - Di_T)
    frame_B = Di_T_B*frame2
    #print(frame_B)
   # print('----------------------------------------------------')
    #cv.imshow('frame', B)   

    B = frame_B + frame_F
    #print(max(B))
    #print(B)
    #print('----------------------------------------------------')

    cv.normalize(B,B_norm,0,1, norm_type = cv.NORM_MINMAX)
    #cv.imshow('frame1', B_norm)
    #cv.waitKey(0)

    frame2_norm = cv.normalize(frame2,0,1,norm_type = cv.NORM_MINMAX)
    DB = np.abs(frame2_norm - B_norm)
    _,DB_bin = cv.threshold(DB, 0.2, 1, type = cv.THRESH_BINARY_INV)
    MO = cv.bitwise_or(DB_bin, B_norm)
    
    opening = cv.morphologyEx(MO, cv.MORPH_OPEN, kernel2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

    #gauss_blur = cv.GaussianBlur(opening, (5,5), 0)

    #cv.imshow('bin', closing)
    closing = np.uint8(closing)
    _, contours,_ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        frame_rgb = cv.drawContours(frame_rgb, [box], 0, (0,0,255), 3)

    cv.imshow('agrvai', frame_rgb)

    #drawing = bdr.boundaries(opening)
    #cv.imshow('bounding', drawing)

    if cv.waitKey(1) & 0xFF == ord('q'):
            break
