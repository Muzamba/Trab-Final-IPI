import cv2 as cv
import numpy as np

cap = cv.VideoCapture('/home/pugdel/Vídeos/aaaa.mp4')
_, frame1 = cap.read()
frame1 = np.float_(frame1)
frame1 = cv.normalize(frame1, 0, 1, norm_type=cv.NORM_MINMAX)


_, frame2 = cap.read()
frame2 = cv.normalize(frame2, 0, 1, norm_type=cv.NORM_MINMAX)
frame2 = np.float_(frame2)
_, frame3 = cap.read()
frame_rgb = frame3
frame3 = np.float_(frame3)
frame3 = cv.normalize(frame3, 0, 1, norm_type=cv.NORM_MINMAX)
Bg = frame3
Thr = np.zeros((frame1.shape[0], frame1.shape[1], 3))
Thr = np.float_(Thr)
Thr = Thr + 0.7
alpha = 0.01

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel2 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 10))
kernel3 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 1))
cont = 0

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

    # blob = np.abs(frame3 - Bg)
    # blob_T = blob > Thr
    # blob = blob * blob_T
    # blob = cv.GaussianBlur(blob, (3, 3), 0)
    # blob = cv.normalize(blob, 0, 255, norm_type=cv.NORM_MINMAX)
    # blob = np.int16(blob)
    # blob = cv.cvtColor(blob, cv.COLOR_BGR2GRAY)
    # _, blob = cv.threshold(blob, 0.5, 1, cv.THRESH_BINARY)

    DB = np.abs(frame3 - Bg)
    _, DB = cv.threshold(DB, 0.1, 1, cv.THRESH_BINARY)
    MO = cv.bitwise_and(DB, Bg)
    a, b, c = cv.split(MO)
    MO = a + b + c
    _, MO = cv.threshold(MO, 0.5, 1, cv.NORM_MINMAX)
    MO = cv.morphologyEx(MO, cv.MORPH_CLOSE, kernel)
    MO = cv.morphologyEx(MO, cv.MORPH_OPEN, kernel1)

    #  MO = cv.erode(MO, kernel3, iterations=1)

    cv.imshow('MO', MO)

    Thr1 = Thr * moving_B
    Thr2 = alpha * Thr * moving_B_N + (1 - alpha) * (5 * np.abs(frame3 - Bg) * moving_B_N)
    Thr = Thr1 + Thr2

    Bg1 = Bg * moving_B
    Bg2 = alpha * Bg * moving_B_N + (1 - alpha) * frame3 * moving_B_N
    Bg = Bg1 + Bg2
    MO = cv.dilate(MO, kernel2)
    MO = cv.normalize(MO, 0, 255, norm_type=cv.NORM_MINMAX)
    MO = np.uint8(MO)
    _, coutours, _ = cv.findContours(MO, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    hsv = cv.cvtColor(frame_rgb, cv.COLOR_BGR2HSV)
    hsv = np.float_(hsv)
    hsv = cv.normalize(hsv, 0, 1, norm_type=cv.NORM_MINMAX)
    frame_aux = np.float_(frame_rgb)
    frame_aux = cv.normalize(frame_aux, 0, 1, norm_type=cv.NORM_MINMAX)
    B, G, R = cv.split(frame_aux)
    I_ = (B + G + R) / 3
    hsv[:, :, 2] = I_
    H_mean = np.mean(hsv[:, :, 0])
    S_mean = np.mean(hsv[:, :, 1])
    I_mean = np.mean(hsv[:, :, 2])
    bottom = np.array([((H_mean - 0.15)/10) - 0.012, ((S_mean + 0.1)/10) - 0.07, (I_mean*1.12) - 0.06])
    upper = np.array([((H_mean - 0.15)/10) + 0.012, ((S_mean + 0.1)/10) + 0.07, (I_mean*1.12) + 0.06])
    teste = cv.inRange(hsv, bottom, upper)
    cv.imshow('teste', teste)
    for count in coutours:
        x, y, w, h = cv.boundingRect(count)

        if 0.231 <= (w / h) <= 0.9:
            cv.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

    print(cont)
    cont += 1

    cv.imshow('aaa', frame_rgb)
    frame3 = np.float_(frame3)
    frame3 = cv.normalize(frame3, 0, 1, norm_type=cv.NORM_MINMAX)

    cv.imshow('Bg', Bg)
    cv.waitKey()
    frame1 = frame2
    frame2 = frame3
    _, frame3 = cap.read()
    frame_rgb = frame3
    frame3 = np.float_(frame3)
    frame3 = cv.normalize(frame3, 0, 1, norm_type=cv.NORM_MINMAX)




  
