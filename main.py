import re
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# GET COLOR MAGNITUDE FROM PIXEL
def getColorMagnitude(pixel):
    return (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))//3

# GET SPECIFIC PIXEL ON (x, y) POSITION FROM THE t-NTH IMAGE
def getPx(imgs, x, y, t):
    return imgs[t].item(y, x)

def getPxMax(imgs, x, y):
    return np.max(imgs[:][0].item(y,x))

def getPxMin(imgs, x, y):
    return np.min(imgs[:][0].item(y,x))

def getPxShadow(imgs, x, y):
    return (int(getPxMax(imgs, x, y)) + int(getPxMin(imgs, x, y)))//2

def getPxDelta(imgs, x, y, t):
    return int(imgs[t].item(x,y) - getPxShadow(imgs, x, y))

# GET SPECIFIC t-NTH IMAGE
def getImg(imgs, t):
    return imgs[t]

# GET MINIMAL AND MAXIMAL COLOR INTENSITIES IN GIVEN PIXEL
# RETURNS NOT ONLY THE INTENSITIES, BUT THE IMAGES IN WHICH THE VALUES WERE RETRIEVED
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

# RETURN TUPLES OF LOCAL DELTA AND TIME
def shadowTime(imgs, x, y):
    deltaTime = []
    for time, img in enumerate(imgs):
        localTime = time
        localDelta =  getPxDelta(imgs, x, y, time)
        deltaTime.append((localDelta, localTime))
    return deltaTime

# RUNS THROUGH EVERY SINGLE FRAME
# CALCULATES DELTA WITH THE IMAGE SHADOW (min+max/2)
# RETURNS THE DELTA OF EVERY SINGLE PIXEL FROM EVERY SINGLE FRAME
# NOT QUITE THERE YET, THE IDEA IS TO INPUT A FRAME AND GET A TIME BACK
def shadowTimeOpt(imgs):
    timeslices = []
    for i, img in enumerate(imgs):
        deltaTime = np.ndarray(img.shape, dtype=np.uint8)
        deltaTime[0:img.shape[0], 0:img.shape[1]] = img[0:img.shape[0], 0:img.shape[1]] - (np.max(img[0:img.shape[0], 0:img.shape[1]]) + np.min(img[0:img.shape[0], 0:img.shape[1]]))//2
        timeslices.append(deltaTime)
    return timeslices

# EVERY FRAME ALL THE TIME
# EACH FRAME WILL HAVE FROM 0 TO 4 INFORMATION:
# 0 INFO -> SHADOW DOESN'T MOVE NOR CHANGE AT ANY GIVEN TIME
# 4 INFO -> SHADOW MOVES IN AND MOVES OUT OF THE FRAME
# THE STORED INFORMATION IS THE TIME IN WHICH STUFF COMES IN AND OUT
# AND THE INTENSITIES OF SAID FRAMES
#[t_in color_in t_out color_out]
#def temporalLocation(imgs, x, y):
#    frames = np.zeros_like(imgs[0])
#    for indexes, img in enumerate(imgs):
#    return frames

# PROGRAM START
path = "./" # Should path be an input?
imgs, calib = [], []
spatial, temporal = [], []
ytop = int(input('enter ytop: '))
ybottom = int(input('enter ybottom: '))
thresh = int(input('enter contrast threshold: '))

# READING FILES IN PATH
for files in os.scandir(path):
    if files.name.rfind(".png"):
        if re.match('lamp_calibration_*', files.name):
            calib.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        elif re.match('000*', files.name):
            imgs.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
# EXTRACTING EDGES FOR POSTERIOR TEMPORAL/LOCATION SHADOW MAPPING
cannyImgs = []
for img in imgs:
    cannyImgs.append(cv.Canny(img,85,255))

# SPATIAL SHADOW LOCATION
# DETECT SHADOW BEHAVIOUR ON PLANE
spatial = spatialLocation(cannyImgs, ytop, ybottom)
print(spatial)

chosenFrames = {}
#for pairing in spatial:
#    if pairing[0] in list(chosenFrames.keys()):
#        continue
#    else:
#        chosenFrames[pairing[0]] = []
#        for i in range(ytop, ybottom):
#            chosenFrames[pairing[0]] += shadowTime(imgs, pairing[0], i)
#for pairing in spatial:
#    if pairing[1] in list(chosenFrames.keys()):
#        continue
#    else:
#        chosenFrames[pairing[1]] = []
#        for i in range(ytop, ybottom):
#            chosenFrames[pairing[1]] += shadowTime(imgs, pairing[1], i)

# TEMPORAL SHADOW LOCATION
# EVERY SINGLE FRAME
#for i in range(imgs[0].shape[0]):
#    for j in range(ytop,ybottom):
#        temporal.append(shadowTime(imgs, range(img.shape[0])[0], range(img.shape[1])[0]))
temporal = shadowTimeOpt(imgs)

#print(chosenFrames)
print(temporal)

# DEBUG
window = plt.figure(figsize=(8,4))
window.add_subplot(131)
plt.imshow(imgs[0])

window.add_subplot(132)
plt.imshow(cannyImgs[0], cmap='gray')

window.add_subplot(133)
plt.imshow(temporal[0], cmap='gray')

plt.show()