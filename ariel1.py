import cv2
import numpy
import cython

cap = cv2.VideoCapture('videos/square.mp4')

_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
_, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
while True:
    first_frame = frame
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    diference = cv2.absdiff(first_gray, gray_frame)
    diference = cv2.dilate(diference,kernal, iterations=10)
    alo = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    _, diference = cv2.threshold(diference, 25, 255, cv2.THRESH_BINARY)
    alo = alo + diference * alo
    #cv2.imshow('teste', frame)
    #cv2.imshow('teste2', first_frame)
    cv2.imshow('teste3', diference)
    #cv2.imshow('teste4', alo)
    #cv2.waitKey()

    #a,contour,b = cv2.findContours(diference.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    '''for c in contour:
        # enter your filtering here
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('box', frame)'''

    key = cv2.waitKey(25)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
