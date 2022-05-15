import re
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# GET COLOR MAGNITUDE FROM PIXEL
def getColorMagnitude(pixel):
    return (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))//3

# GET SPECIFIC PIXEL ON (x, y) POSITION FROM THE t-NTH IMAGE
def getPx(imgs, x, y, t):
    return imgs[t][y][x]

def getPxMax(imgs, x, y):
    return np.max(imgs[:][0][y][x])

def getPxMin(imgs, x, y):
    return np.min(imgs[:][0][y][x])

def getPxShadow(imgs, x, y):
    return (getPxMax(imgs, x, y) + getPxMin(imgs, x, y))/2

def getPxDelta(imgs, x, y, t):
    return getPx(imgs, x, y, t) - getPxShadow(imgs, x, y)

# GET SPECIFIC t-NTH IMAGE
def getImg(imgs, t):
    return imgs[t]

# GET MINIMAL AND MAXIMAL COLOR INTENSITIES IN GIVEN PIXEL
# RETURNS NOT ONLY THE INTENSITIES, BUT THE IMAGES IN WHICH THE VALUES WERE RETRIEVED
def getPixelMinMax(imgs, x, y):
    minVal, maxVal = 256, 0
    minIndex, maxIndex = 0, 0,
    for indexes, img in enumerate(imgs[0:-2]):
        if img[y][x] < minVal:
            minVal = img[y][x]
            minIndex = indexes
        if img[y][x] > maxVal:
            maxVal = img[y][x]
            maxIndex = indexes
    return minVal, minIndex, maxVal, maxIndex

# GET THE MAXIMAL INTENSITY VALUE,
# AND THE IMAGE IN WHICH THE RETURNED VALUE WAS FOUND
#def getMaxMag(imgs):
#    maxMag, maxIndex = 0, 0
#    for indexes, img in enumerate(imgs[0:-2]):
#        if(np.max(img.ravel()) > maxMag):
#            maxMag = np.max(img)
#            maxIndex = indexes
#    return maxMag, maxIndex

# GET THE MINIMAL INTENSITY VALUE,
# AND THE IMAGE IN WHICH THE RETURNED VALUE WAS FOUND
#def getMinMag(imgs):
#    minMag, minIndex = 0, 0
#    for indexes, img in enumerate(imgs[0:-2]):
#        if(np.min(img.ravel()) < minMag):
#            minMag = np.min(img)
#            minIndex = indexes
#    return minMag, minIndex

# GIVEN A SET OF CONTOUR IMAGES
# AND A PAIR OF TOP AND BOTTOM Y POSITIONS
# WHERE, IN THE X-AXIS, IS LOCATED THE SHADOW EDGE?
def spatialLocation(imgs, ytop, ybottom):
    spatial = []
    for img in imgs[0:-2]:
        topContour = np.nonzero(img[:][ytop])[0].ravel()[0]
        bottomContour = np.nonzero(img[:][ybottom])[0].ravel()[0]
        spatial.append((topContour, bottomContour))
    return spatial

# EVERY FRAME ALL THE TIME
# EACH FRAME WILL HAVE FROM 0 TO 4 INFORMATION:
# 0 INFO -> SHADOW DOESN'T MOVE NOR CHANGE AT ANY GIVEN TIME
# 4 INFO -> SHADOW MOVES IN AND MOVES OUT OF THE FRAME
# THE STORED INFORMATION IS THE TIME IN WHICH STUFF COMES IN AND OUT
# AND THE INTENSITIES OF SAID FRAMES
#[t_in color_in t_out color_out]
def temporalLocation(imgs):
    frames = np.zeros_like(imgs[0])
    for img in imgs:
        print("ablublublue")
    return frames

path = "./" # Should path be an input?
imgs, calib = [], []
spatial, temporal = [], []
ytop = int(input('enter ytop: '))
ybottom = int(input('enter ybottom: '))

# READING FILES IN PATH
for files in os.scandir(path):
    if files.name.rfind(".png"):
        if re.match('lamp_calibration_*', files.name):
            calib.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        else:
            imgs.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))

# EXTRACTING EDGES FOR POSTERIOR TEMPORAL/LOCATION SHADOW MAPPING
cannyImgs = []
for img in imgs:
    cannyImgs.append(cv.Canny(img,37,222))

# SPATIAL SHADOW LOCATION
# DETECT SHADOW BEHAVIOUR ON PLANE
spatial = spatialLocation(cannyImgs, ytop, ybottom)
print(str(spatial))

for i in range(len(imgs)-2):
    print(str(getPx(imgs, 703, 914, i)))

# DEBUG
window = plt.figure(figsize=(8,4))
window.add_subplot(121)
plt.imshow(imgs[0])

window.add_subplot(122)
plt.imshow(cannyImgs[0], cmap='gray')

plt.show()