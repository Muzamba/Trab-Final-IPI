import numpy as np
import cv2 as cv
import boundinrect as bdr

cap = cv.VideoCapture('videos/rain.mp4')
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
    merge = np.zeros((frame.shape[0], frame.shape[1], 3))

    frame_hsi = cv.cvtColor(frame_rgb, cv.COLOR_BGR2HSV)
    frame_hsi[:,:,2] = (1/3)*(frame_rgb[:,:,0] + frame_rgb[:,:,1] + frame_rgb[:,:,2]) 
    frame_mean = frame.mean()

    H_top = (frame_mean-0.15)/10 + 0.012
    H_bottom = (frame_mean-0.15)/10 - 0.012
    S_top = (frame_mean+0.1)/10 + 0.07
    S_bottom = (frame_mean+0.1)/10 - 0.07
    I_top = (1.12*frame_mean) + 0.06
    I_bottom = (1.12*frame_mean) - 0.06

    boolH_top = frame_hsi[:,:,0] <= H_top
    boolH_bottom = frame_hsi[:,:,0] >= H_bottom
    boolH = boolH_bottom & boolH_top

    boolS_top = frame_hsi[:,:,1] <= S_top 
    boolS_bottom = frame_hsi[:,:,1] >= S_bottom
    boolS = boolS_bottom & boolS_top        

    boolI_top = frame_hsi[:,:,2] <= I_top    
    boolI_bottom = frame_hsi[:,:,2] >= I_bottom
    boolI =  boolI_bottom & boolI_top

    boolMask = boolH & boolS & boolI
    merge[:,:,0] = boolMask
    merge[:,:,1] = boolMask
    merge[:,:,2] = boolMask

    AAI = merge*frame_hsi
    cv.imshow('aai', AAI)

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
    '''_, contours,_ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        frame_rgb = cv.drawContours(frame_rgb, [box], 0, (0,0,255), 3)

    cv.imshow('agrvai', frame_rgb)'''

    #drawing = bdr.boundaries(opening)
    #cv.imshow('bounding', drawing)

    if cv.waitKey(1) & 0xFF == ord('q'):
            break
