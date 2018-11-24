import cv2 as cv
import numpy as np

cap = cv.VideoCapture('videos/square.mp4')
_,frame = cap.read()
#out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc(*'DIVX'), 25.0, (frame.shape[0],frame.shape[1]))

_, frame2_n = cap.read()
_, frame1_n = cap.read()
_, frame0 = cap.read()
_, frame1_p = cap.read()
_, frame2_p = cap.read()


kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
kernel2 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))

while cap.isOpened():
    frame2_n = frame1_n
    frame1_n = frame0
    frame0 = frame1_p
    frame1_p = frame2_p
    _, frame2_p = cap.read()

    B2n, G2n, R2n = cv.split(frame2_n)
    B2p, G2p, R2p = cv.split(frame2_p)
    B1n, G1n, R1n = cv.split(frame1_n)
    B1p, G1p, R1p = cv.split(frame1_p)
    B2, G2, R2 = cv.split(frame0)

    test = (B2 + G2 + R2) / 3
    test1_n = (B1n + G1n + R1n) / 3
    test1_p = (B1p + G1p + R1p) / 3
    test2_n = (B2n + G2n + R2n) / 3
    test2_p = (B2p + G2p + R2p) / 3

    diff1_n = abs(test1_n - test)
    diff1_p = abs(test1_p - test)
    diff2_n = abs(test2_n - test)
    diff2_p = abs(test2_p - test)

    _, diff1_n = cv.threshold(diff1_n, 5, np.max(diff1_n), cv.THRESH_BINARY)
    _, diff1_p = cv.threshold(diff1_p, 5, np.max(diff1_p), cv.THRESH_BINARY)
    _, diff2_n = cv.threshold(diff2_n, 5, np.max(diff2_n), cv.THRESH_BINARY)
    _, diff2_p = cv.threshold(diff2_p, 5, np.max(diff2_p), cv.THRESH_BINARY)

    LAO2 = cv.bitwise_and(diff2_n, diff2_p)
    LAO1 = cv.bitwise_and(diff1_n, diff1_p)

    final = cv.bitwise_or(LAO1, LAO2)
    final = cv.morphologyEx(final, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(final, cv.MORPH_CLOSE, kernel2)

    #cv.imshow('diff2_n', diff2_n)
    #cv.imshow('diff2_p', diff2_p)
    #cv.imshow('diff1_n', diff1_n)
    #cv.imshow('diff1_p', diff1_p)
    cv.imshow('final', closing)



    '''
    opening = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)
    cv.imshow('1', frame2_n)
    cv.imshow('2', frame1_n)
    cv.imshow('3', frame0)
    cv.imshow('4 ', frame1_p)
    cv.imshow('5 ', frame2_p)'''

    
    if cv.waitKey(1) & 0xFF == ord('q'):
            break

#out.release()
cap.release()