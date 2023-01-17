import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

red = np.uint8([[[0,0,255 ]]])
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
print(hsv_red)
thickness = 5
img = np.zeros((512,512,3), np.uint8)
img = cv2.rectangle(img, (0,0), (511,511), np.array([255,0,0]), -1)

cv2.imshow("lalala", img)
k = cv2.waitKey(0) # 0==wait forever