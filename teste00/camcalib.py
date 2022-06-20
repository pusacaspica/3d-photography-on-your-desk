import sys
import re
import os
import cv2 as cv
from cv2 import projectPoints
import numpy as np
import matplotlib.pyplot as plt

# USING THE LAMP CALIBRATION IMAGE
imgPoints = np.float32([[499, 81, 1],[534, 36, 1],
            [168, 40,1],[168, 69,1],
            [489, 930,1],[515, 971,1],
            [163, 865,1],[163, 892,1]])
objPoints = np.float32([[470.448-397.034, 241.187 + 229.608, 70.0,1],[397.034-397.034, 231.625 + 229.608, 0.0,1],
            [752.245-397.034, 307.869 + 229.608, 70.0,1],[667.416-397.034, 295.374 + 229.608, 0.0,1],
            [479.73-397.034, -220.556 + 229.608, 70.0,1],[406.089-397.034, -211.845 + 229.608, 0.0,1],
            [753.028-397.034, -229.608 + 229.608, 70.0,1], [667.474-397.034, -220.295 + 229.608, 0.0,1]])
            #MAX[0]: 397.034; MIN[1]: -229.608

img = cv.imread("lamp_calibration_00.jpg", cv.IMREAD_GRAYSCALE)
img = np.array(img)

# NORMALIZATION

height, width = img.shape

# normalized image coordinates
xBounds = [np.divide(np.min(imgPoints, axis=0), np.kron(np.ones((2,1)),imgPoints)), np.divide(np.max(imgPoints, axis=0), np.kron(np.ones((2,1)),imgPoints))]
T = np.transpose(np.array([[width + height, 0, 0], [0, width+height, 0], [width/2, height/2, 1]]))
Tinv = np.linalg.inv(T)
xTild = []
for point in imgPoints:
    xTild.append(np.matmul(Tinv, point))
xTild = np.array(xTild)

Height = 537.477
Width = 355.211
U = np.transpose(np.array([[Width + Height + 1, 0, 0, 0], [0, Width+Height+1, 0, 0], [0, 0, Width+Height+1, 0], [Width/2, Height/2, 1/2, 1]]))
print(U)
Uinv = np.linalg.inv(U)
XTild = []
for point in objPoints:
    XTild.append(np.matmul(Uinv, point))
XTild = np.array(XTild)
print(XTild)

# DLT
print(imgPoints.shape)
n = imgPoints.shape[1]
A = np.zeros((2*n, 12))
for i in range(2*n):
    A[i] = [0, 0, 0, 0, np.transpose(XTild)[i], xTild[i, 2] * np.transpose(XTild)[i], np.transpose(XTild)[i], 0, 0, 0, 0, -xTild[i, 1]*np.transpose(XTild)[i]]
print(A)
print(A.shape)