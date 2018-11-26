import cv2
import numpy as np

cap = cv2.VideoCapture('videos/rain.mp4')
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows = False)
kernel_o = np.ones((10,10), np.uint8)
kernel_c = np.ones((7,7), np.uint8)
kernel_e = np.ones((5,5), np.uint8) 


if (cap.isOpened()== False): 
  print("Error opening video stream or file")

video = []
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
 
    video.append(frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  else: 
    break

for frame in video:
    fgmask = fgbg.apply(frame)
    new_fgmask = np.zeros((frame.shape), dtype = np.uint8)
    new_fgmask[:,:,0] = fgmask[:,:]
    new_fgmask[:,:,1] = fgmask[:,:]
    new_fgmask[:,:,2] = fgmask[:,:]
    sla = new_fgmask * frame
    rgb_sla = cv2.cvtColor(sla, cv2.COLOR_HSV2RGB)
    gray_sla = cv2.cvtColor(rgb_sla, cv2.COLOR_RGB2GRAY)
    ret, bin_sla = cv2.threshold(gray_sla, 127, 255, cv2.THRESH_BINARY)
    mean = frame.mean()
    H_top = (mean-0.15)/10 + 0.012
    H_bottom = (mean-0.15)/10 - 0.012
    S_top = (mean+0.1)/10 + 0.07
    S_bottom = (mean+0.1)/10 - 0.07
    I_top = (1.12*mean) + 0.06
    I_bottom = (1.12*mean) - 0.06

    B,G,R = cv2.split(frame)
    frame_hsi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsi[:,:,2] = (1/3)*(R+G+B)

    cv2.imshow('frame', bin_sla)
    cv2.waitKey()

    boolH_top = frame_hsi[:,:,0] <= H_top
    boolH_bottom = frame_hsi[:,:,0] >= H_bottom

    boolS_top = frame_hsi[:,:,1] <= S_top 
    boolS_bottom = frame_hsi[:,:,1] >= S_bottom          

    boolI_top = frame_hsi[:,:,2] <= I_top    
    boolI_bottom = frame_hsi[:,:,2] >= I_bottom

    
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_c)
    
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_o)
    #erosion = cv2.erode(opening,kernel_e,iterations = 1)
    #print(opening.shape, frame.shape)
    #new_frame = cv2.absdiff(gray_frame, opening)
    cv2.imshow('r', opening)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()