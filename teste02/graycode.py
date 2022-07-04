from email.mime import image
import PIL as pl
import sys
import re
import os
import cv2 as cv
from cv2 import projectPoints
from cv2 import CALIB_USE_INTRINSIC_GUESS
from cv2.structured_light import GrayCodePattern
import json
import numpy
import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.plot_utils as pp
import pytransform3d.transformations as pts
import pytransform3d.rotations as pr
import matplotlib.pyplot as plt

# PROGRAM START
the_path = "./" # TRIED MAKING IT AN INPUT, DIDN'T WORK
imgs, calibImgs, lampcalib, camcalib = [], [], [], []
maxs,  mins = [], []
spatial, temporal = [], []
ytop = int(input('enter ytop: '))
ybottom = int(input('enter ybottom: '))
thresh = int(input('enter contrast threshold: '))

# READING FILES IN PATH
for files in os.scandir(the_path):
    if files.name.rfind(".jpg"):
        if re.match('lamp_calibration_*', files.name):
            #print(files.name)
            lampcalib.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        elif re.match('00*', files.name):
            #print(files.name)
            imgs.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        elif re.match('camera_calibration_*', files.name):
            print("aye")
            camcalib.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        else:
            continue

# CONVERTING IMGS TO NP.ARRAY
imgs = np.array(imgs)
lampcalib = np.array(lampcalib)
camcalib = np.array(camcalib)

# CAMERA CALIBRATION
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints =[] 
imgpoints = []

ret, corners = cv.findChessboardCorners(camcalib[0], (7, 6), None)
print(corners.ravel())
centerImgOffset = np.array([imgs[0].shape[0], imgs[0].shape[1]])/2 - np.array([corners.ravel()[0], corners.ravel()[1]]) 
print(centerImgOffset)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(camcalib[0], corners, (11,11), (-1, -1), criteria)
    imgpoints.append(corners2)

corners = np.array(corners)
corners = corners.reshape(corners.shape[0], corners.shape[2])
objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)
imgpoints = imgpoints.reshape(imgpoints.shape[2], imgpoints.shape[1],imgpoints.shape[3])
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, camcalib[0].shape[::-1], None, None)
rmat,jacobian = cv.Rodrigues(rvecs[0])

h, w = camcalib[0].shape[:2]
optMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
x, y, w, h = roi

for i, img in enumerate(imgs):
    calibrated = cv.undistort(img, mtx, dist, None, optMtx)
    calibImgs.append(calibrated[y:y+h, x:x+w])
calibImgs = np.array(calibImgs)

pattern = GrayCodePattern.create(calibImgs[0].shape[0], calibImgs[0].shape[1])

whiteimg = np.ones((calibImgs[0].shape[0], calibImgs[0].shape[1]), dtype = np.uint8) * np.array([255])
white = pl.Image.fromarray(whiteimg)
blackimg = np.zeros((calibImgs[0].shape[0], calibImgs[0].shape[1]), dtype = np.uint8)
black = pl.Image.fromarray(blackimg)
print(blackimg)

blackmask, whitemask = GrayCodePattern.getImagesForShadowMasks(black, white)

print(GrayCodePattern.getNumberOfPatternImages())