import cv2
import  numpy as np

cap = cv2.VideoCapture('videos/people.mp4')

_, frame = cap.read()
bg = frame * 0
'''bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)'''
while True:
    frame2 = frame
    _, frame = cap.read()

    diff = cv2.absdiff(frame, frame2)
    diff = diff * 0.01
    bg = bg + diff

    cv2.imshow('teste',bg)
    #cv2.waitKey()
    key = cv2.waitKey(30)
    if key == 27:
        break
