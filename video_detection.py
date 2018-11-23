import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('videos/people.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel_o = np.ones((10,10), np.uint8)
kernel_c = np.ones((7,7), np.uint8)
kernel_e = np.ones((5,5), np.uint8) 
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

video = []
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    video.append(frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  else: 
    break

for frame in video:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    fgmask = fgbg.apply(gray_frame)
    
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_c)
    
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_o)
    #erosion = cv2.erode(opening,kernel_e,iterations = 1)
    #print(opening.shape, frame.shape)
    new_frame = cv2.absdiff(gray_frame, opening)
    cv2.imshow('r', new_frame)
    cv2.waitKey(10)












# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()