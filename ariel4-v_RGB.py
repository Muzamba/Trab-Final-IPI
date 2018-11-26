import cv2 as cv
import numpy as np

cap = cv.VideoCapture('/home/pugdel/Vídeos/people-walking.mp4')

_, frame1 = cap.read()
frame1 = np.float_(frame1)
frame1 = cv.normalize(frame1, 0, 1, norm_type=cv.NORM_MINMAX)


_, frame2 = cap.read()
frame2 = cv.normalize(frame2, 0, 1, norm_type=cv.NORM_MINMAX)
frame2 = np.float_(frame2)
_, frame3 = cap.read()
frame3 = cv.normalize(frame3, 0, 1, norm_type=cv.NORM_MINMAX)
frame3 = np.float_(frame3)
Bg = frame3
Thr = np.zeros((frame1.shape[0], frame1.shape[1], 3))
Thr = np.float_(Thr)
Thr = Thr =+ 0.5
alpha = 0.0001

while True:
    D1 = np.abs(frame3 - frame2)
    D2 = np.abs(frame3 - frame1)

    D1_B = D1 > Thr
    D1_T = D1_B * frame3
    D2_B = D2 > Thr
    D2_T = D2_B * frame3
    moving = cv.bitwise_and(D1_T, D2_T)
    moving_B = D1_B & D2_B
    moving_B_N = 1 - moving_B

    blob = np.abs(frame3 - Bg)
    blob_T = blob > Thr
    blob = blob * blob_T
    blob = cv.GaussianBlur(blob, (3, 3), 0)
    # blob = cv.normalize(blob, 0, 255, norm_type=cv.NORM_MINMAX)
    # blob = np.int16(blob)
    # blob = cv.cvtColor(blob, cv.COLOR_BGR2GRAY)
    # _, blob = cv.threshold(blob, 0.5, 1, cv.THRESH_BINARY)

    DB = np.abs(frame3 - Bg)
    _, DB = cv.threshold(DB, 0.1, 1, cv.THRESH_BINARY)
    MO = cv.bitwise_and(DB, Bg)
    cv.imshow('MO', MO)

    Thr1 = Thr * moving_B
    Thr2 = alpha * Thr * moving_B_N + (1 - alpha) * (5 * np.abs(frame3 - Bg) * moving_B_N)
    Thr = Thr1 + Thr2

    Bg1 = Bg * moving_B
    Bg2 = alpha * Bg * moving_B_N + (1 - alpha) * frame3 * moving_B_N
    Bg = Bg1 + Bg2

    cv.imshow('blob', blob)
    cv.imshow('Bg', Bg)
    cv.waitKey()
    frame1 = frame2
    frame2 = frame3
    _, frame3 = cap.read()
    frame3 = np.float_(frame3)
    frame3 = cv.normalize(frame3, 0, 1, norm_type=cv.NORM_MINMAX)


