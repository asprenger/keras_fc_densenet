import numpy as np
import cv2

# (720, 960, 3)
# 3751 frames

cap = cv2.VideoCapture('01TP_extract.avi')
nb_frames = 0
while(cap.isOpened()):
    ret_val, frame = cap.read()
    if not ret_val:
        break
    nb_frames += 1
    resized = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    print('%d: %s' % (nb_frames, resized.shape))
cap.release()