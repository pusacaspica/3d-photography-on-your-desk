import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# print(objp)

objpoints =[] 
imgpoints = []

image = cv.imread("camera_calibration_0.jpg", cv.IMREAD_GRAYSCALE)

ret, corners = cv.findChessboardCorners(image, (7, 6), None)
print(ret)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(image, corners, (11,11), (-1, -1), criteria)
    imgpoints.append(corners)
    '''cv.drawChessboardCorners(image, (7, 6), corners2, ret)
    plt.imshow(image)
    plt.show()'''
corners = np.array(corners)
corners = corners.reshape(corners.shape[0], corners.shape[2])
print(corners.shape)
objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)
print(imgpoints.shape)
imgpoints = imgpoints.reshape(imgpoints.shape[2], imgpoints.shape[1],imgpoints.shape[3])
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image.shape[::-1], None, None)
h, w = image.shape[:2]
uncalib = cv.imread("0075.jpg", cv.IMREAD_GRAYSCALE)
newMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
rawDst = cv.undistort(uncalib, mtx, dist, None, newMtx)
print(roi)
x, y, w, h = roi
dst = rawDst[y:y+h, x:x+w]
plt.subplot(221)
plt.imshow(uncalib)
plt.subplot(223)
plt.imshow(rawDst)
plt.subplot(224)
plt.imshow(dst)
plt.show()