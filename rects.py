import cv2 as cv
import numpy as np

img = cv.imread('images/www.png')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, bin_img = cv.threshold(gray_img, 128, 1, cv.THRESH_BINARY)

_, contours,_ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#print(contours[0])

print('Number of shapes {}'.format(len(contours)))

for contour in contours:
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    box_q = box
    img = cv.drawContours(img, [box], 0, (0,0,255), 3)

#img = cv.drawContours(img,[[[54 425], ]])
#print(box_q)
#box_q = [[755 ,185],[611, 110], [670, -2], [814, 72]]
#img = cv.drawContours(img, [box], 0, (2,254,213), 3)

cv.imshow('new', img)
cv.waitKey(0)
cv.destroyAllWindows()