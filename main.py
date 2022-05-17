import sys
import re
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

# GET COLOR MAGNITUDE FROM PIXEL
def getColorMagnitude(pixel):
    return (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))//3

# GET SPECIFIC PIXEL ON (x, y) POSITION FROM THE t-NTH IMAGE
def getPx(imgs, x, y, t):
    return imgs[t].item(y, x)

def getPxMax(imgs, x, y):
    #debug = np.ndarray(imgs[0].shape, dtype=np.uint8)
    #debugre = np.array([debug])
    #debugre[0, 0:imgs[0].shape[0], 0:imgs[0].shape[1]] = np.max(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])
    #print(len(debugre[0]))
    return np.max(imgs[:][0].item(y,x))

def getPxMin(imgs, x, y):
    #np.min(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])
    return np.min(imgs[:][0].item(y,x))

def getPxShadow(imgs, x, y):
    return (int(getPxMax(imgs, x, y)) + int(getPxMin(imgs, x, y)))//2

def getPxDelta(imgs, x, y, t):
    return int(imgs[t].item(y,x) - getPxShadow(imgs, x, y))

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
        #print(topContour)
        bottomContour = np.nonzero(img[:][ybottom])[0].ravel()[0]
        #print(bottomContour)
        spatial.append((topContour, bottomContour))
    return spatial

# RETURN TUPLES OF LOCAL DELTA AND TIME IN SINGLE FRAME
def shadowTime(imgs, x, y, thresh):
    deltaTime = -1
    for i, img in enumerate(imgs):
        if (abs(img.item(y,x) - ((getPxMin(imgs, x, y) + getPxMax(imgs, x, y))//2))) > thresh:
            if deltaTime == -1:
                deltaTime = i
                break
    return int(deltaTime)

# RUNS THROUGH EVERY SINGLE FRAME
# CALCULATES DELTA WITH THE IMAGE SHADOW (min+max/2)
# RETURNS THE DELTA OF EVERY SINGLE PIXEL FROM EVERY SINGLE FRAME
# NOT QUITE THERE YET, THE IDEA IS TO INPUT A FRAME AND GET A TIME BACK
def shadowTimeOpt(imgs):
    timeslices = []
    for img in imgs:
        deltaTime = np.ndarray(img.shape, dtype=np.uint8)
        deltaTime[0:img.shape[0], 0:img.shape[1]] = img[0:img.shape[0], 0:img.shape[1]] - (np.max(img[0:img.shape[0], 0:img.shape[1]]) + np.min(img[0:img.shape[0], 0:img.shape[1]]))//2
        timeslices.append(deltaTime)
    return timeslices

# PAINTS EVERY FRAME TAKING PIXEL DEPTH IN CONSIDERATION
# IF IT PAINTS IN bottom, FRAME DOESN'T CONTRIBUTE TO DEPTH
def shadowTimeImg(imgs, thresh, bottom):
    timeslice = np.ndarray(imgs[0].shape, dtype=np.uint8)
    timeslices = np.array([timeslice]*50)
    timeslices[0:len(imgs), 0:imgs[0].shape[0], 0:imgs[0].shape[1]] = np.where(abs(np.array(imgs[0:len(imgs)])[0:imgs[0].shape[0], 0:imgs[0].shape[1]] - (np.max(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])]) + np.min(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])//2)) > thresh, np.array(imgs[0:len(imgs)])[0:imgs[0].shape[0], 0:imgs[0].shape[1]] - (np.max(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])]) + np.min(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])//[2])[0], bottom)
    return timeslices[1]

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
            print(files.name)
            imgs.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        else:
            continue
# EXTRACTING EDGES FOR POSTERIOR TEMPORAL/LOCATION SHADOW MAPPING
cannyImgs = []
for img in imgs:
    cannyImgs.append(cv.Canny(img,85,255))
cannyImgs = np.array(cannyImgs)

# SPATIAL SHADOW LOCATION
# DETECT SHADOW BEHAVIOUR ON PLANE
spatial = spatialLocation(cannyImgs, ytop, ybottom)
#print(spatial)

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
temporal = shadowTimeImg(imgs, thresh, -5)
#print(chosenFrames)
#print(temporal)

#f = open("temporal.txt", "a")
#f.write(str(temporal))
#f.close()

#allFrames = shadowTimeImg(imgs, thresh)
#print(allFrames)
#getPxMax(imgs, 0, 0)

# DEBUG
window = plt.figure(figsize=(8,4))
window.add_subplot(131)
plt.imshow(imgs[0])

window.add_subplot(132)
plt.imshow(cannyImgs[0], cmap='gray')

window.add_subplot(133)
plt.imshow(temporal, cmap='gray')

plt.show()