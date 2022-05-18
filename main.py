import sys
import re
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=sys.maxsize)

# GET COLOR MAGNITUDE FROM PIXEL
def getColorMagnitude(pixel):
    return (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))//3

# GET SPECIFIC PIXEL ON (x, y) POSITION FROM THE t-NTH IMAGE
def getPx(imgs, x, y, t):
    return imgs[t].item(y, x)

def getPxMax(imgs, x, y):
    #np.max(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])
    return np.max(imgs[:][0].item(y,x))

def getPxMin(imgs, x, y):
    #np.min(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])
    return np.min(imgs[:][0].item(y,x))

def getPxShadow(imgs, x, y):
    return (int(getPxMax(imgs, x, y)) + int(getPxMin(imgs, x, y)))//2

def getPxDelta(imgs, x, y, t):
    return int(imgs[t].item(y,x) - getPxShadow(imgs, x, y))

# GET MINIMAL AND MAXIMAL COLOR INTENSITIES IN GIVEN PIXEL
# RETURNS NOT ONLY THE INTENSITIES, BUT THE IMAGES IN WHICH THE VALUES WERE RETRIEVED
# I'm not sure why I wrote this one right here
def getPixelMinMax(imgs, x, y):
    minVal, maxVal = 256, 0
    minIndex, maxIndex = 0, 0,
    for indexes, img in enumerate(imgs):
        if img[x][y] < minVal:
            minVal = img[x][y]
            minIndex = indexes
        if img[x][y] > maxVal:
            maxVal = img[x][y]
            maxIndex = indexes
    return minVal, minIndex, maxVal, maxIndex

# GIVEN A SET OF CONTOUR IMAGES
# AND A PAIR OF TOP AND BOTTOM Y POSITIONS
# WHERE, IN THE X-AXIS, IS LOCATED THE SHADOW EDGE?
def spatialLocation(imgs, ytop, ybottom):
    spatial = []
    for img in imgs:
        topContour = np.nonzero(img[:][ytop])[0].ravel()[0]
        bottomContour = np.nonzero(img[:][ybottom])[0].ravel()[0]
        spatial.append((topContour, bottomContour))
    return spatial

# RETURN FRAME IN WHICH THE FRAME IS REACHED
def shadowTime(imgs, x, y, thresh):
    if x == imgs[0].shape[1]:
        return -1
    for i, img in enumerate(imgs):
        if (abs(img.item(y,x) - ((getPxMin(imgs, x, y) + getPxMax(imgs, x, y))//2))) > thresh:
            return i
    return shadowTime(imgs, x+1, y, thresh)

# PAINTS EVERY FRAME TAKING PIXEL DEPTH IN CONSIDERATION; IF IT PAINTS IN bottom, FRAME DOESN'T CONTRIBUTE TO DEPTH
# PRINTS A NEW IMAGE TO BE PLOTTED
# USED ONLY FOR DEBUGGING, NOT MEANT TO WORK WITHIN THE ALGORITHM DESCRIBED.
def imageDelta(imgs, thresh, bottom):
    timeslice = np.zeros(imgs[0].shape, dtype=np.uint8)
    timeslices = np.array([timeslice]*50)
    # My life has been forever changed by the next line
    timeslices[0:len(imgs), 0:imgs[0].shape[0], 0:imgs[0].shape[1]] = np.where(abs(imgs[0:len(imgs)][0:imgs[0].shape[0], 0:imgs[0].shape[1]] - (np.max(imgs[0:len(imgs)][0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])]) + np.min(imgs[0:len(imgs)][0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])//2)) > thresh, imgs[0:len(imgs)][0:imgs[0].shape[0], 0:imgs[0].shape[1]] - (np.max(imgs[0:len(imgs)][0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])]) + np.min(imgs[0:len(imgs)][0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])//2), np.uint8(bottom))
    return timeslices

# PROGRAM START
path = "./" # Should path be an input?
imgs, calib = [], []
maxs,  mins = [], []
spatial, temporal = [], []
ytop = int(input('enter ytop: '))
ybottom = int(input('enter ybottom: '))
thresh = int(input('enter contrast threshold: '))

# READING FILES IN PATH
for files in os.scandir(path):
    if files.name.rfind(".png"):
        if re.match('lamp_calibration_*', files.name):
            #print(files.name)
            calib.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        elif re.match('00*', files.name):
            #print(files.name)
            imgs.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        else:
            continue
# EXTRACTING EDGES FOR POSTERIOR SPATIAL SHADOW MAPPING
cannyImgs = []
for img in imgs:
    cannyImgs.append(cv.Canny(img,85,255))
cannyImgs = np.array(cannyImgs)

# SPATIAL SHADOW LOCATION
# DETECT SHADOW BEHAVIOUR ON PLANE
spatial = spatialLocation(cannyImgs, ytop, ybottom)
print(spatial)

time = shadowTime(imgs, 200, 13, thresh)
print(time)

# DEBUG
window = plt.figure(figsize=(8,4))
window.add_subplot(131)
plt.imshow(imgs[0])

window.add_subplot(132)
plt.imshow(cannyImgs[0], cmap='gray')

#window.add_subplot(133)
#plt.imshow(deltas[1], cmap='gray')

plt.show()