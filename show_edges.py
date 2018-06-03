import skimage.feature
import numpy as np
import cv2

cam = cv2.VideoCapture(0)

while True:
    ret_val, img = cam.read()
    image = skimage.feature.canny(image=img)
    cv2.imshow('Edges test', image)
    if cv2.waitKey(1) == 27:
        break
