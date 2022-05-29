import sys
import re
import os
import cv2 as cv
from cv2 import projectPoints
import numpy as np
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=sys.maxsize)

# GET COLOR MAGNITUDE FROM PIXEL
def getColorMagnitude(pixel):
    return (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))//3

def getPxMax(imgs, x, y):
    #np.max(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])
    return np.max(imgs[:][0].item(y,x))

def getPxMin(imgs, x, y):
    #np.min(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])
    return np.min(imgs[:][0].item(y,x))

def getPxShadow(imgs, x, y):
    return (int(getPxMax(imgs, x, y)) + int(getPxMin(imgs, x, y)))//2

def delta(imgs, img, x, y):
    return abs(img[y, x] - ((getPxMin(imgs, x, y) + getPxMax(imgs, x, y))//2))

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


# RETURN FRAME IN WHICH THE FRAME IS REACHED BY SHADOW
# TIME -1 MEANS THAT IT IS OUT OF BOUNDS OF THE SCAN
def shadowTime(imgs, x, y, thresh):
    if int(x) == imgs[0].shape[1]:
        return -1
    for i, img in enumerate(imgs):
        if (delta(imgs, img, x, y)) > thresh:
            if i == (len(imgs)-1):
                return -1
            else:
                return i
    return shadowTime(imgs, x+1, y, thresh)

""" def optShadowTime(imgs, thresh):
    times = np.zeros(imgs[0].shape)
    for i, img in enumerate(imgs):
        if delta(imgs, img, np.arange(0, img.shape[0], 1), np.arange(0, img.shape[1], 1)) > thresh:
            times[0:img.shape[0], 0:img.shape[1]] = i """
        
# PAINTS EVERY FRAME TAKING PIXEL DEPTH IN CONSIDERATION; IF IT PAINTS IN bottom, FRAME DOESN'T CONTRIBUTE TO DEPTH
# PRINTS A NEW IMAGE TO BE PLOTTED
# USED ONLY FOR DEBUGGING, NOT MEANT TO WORK WITHIN THE ALGORITHM DESCRIBED.
def imageDelta(imgs, thresh, bottom):
    timeslice = np.zeros(imgs[0].shape, dtype=np.uint8)
    timeslices = np.array([timeslice]*50)
    # My life has been forever changed by the next line
    timeslices[0:len(imgs), 0:imgs[0].shape[0], 0:imgs[0].shape[1]] = np.where(abs(np.array(imgs[0:len(imgs)])[0:imgs[0].shape[0], 0:imgs[0].shape[1]] - (np.max(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])]) + np.min(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])//2)) > thresh, np.array(imgs[0:len(imgs)])[0:imgs[0].shape[0], 0:imgs[0].shape[1]] - (np.max(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])]) + np.min(np.array(imgs[0:len(imgs)])[0:int(imgs[0].shape[0]), 0:int(imgs[0].shape[1])])//[2])[0], np.uint8(bottom))
    return timeslices

def paintY(img, x):
    img[x, 0:img.shape[1]] = 125

def paintX(img, x):
    img[0:img.shape[0], x] = 125
    
# OPTIMIZED VERSIONS OF THE METHODS ABOVE
# Not working yet
# Crucial if we're going to do this
def optMin(imgs):
    ret = index = np.ndarray((imgs[0].shape[0], imgs[0].shape[1]))
    ret[0:ret.shape[0], 0:ret.shape[1]] = np.min(imgs[0:imgs[0].shape[0], 0:imgs[0].shape[1]])
    index[0:ret.shape[0], 0:ret.shape[1]] = np.argmin(imgs[0:imgs[0].shape[0], 0:imgs[0].shape[1]], axis=0)
    return ret, index

def optMax(imgs):
    ret = index = np.ndarray((imgs[0].shape[0], imgs[0].shape[1]))
    ret[0:ret.shape[0], 0:ret.shape[1]] = np.max(imgs[0:imgs[0].shape[0], 0:imgs[0].shape[1]])
    index[0:ret.shape[0], 0:ret.shape[1]] = np.argmax(imgs[:, 0:imgs[0].shape[0], 0:imgs[0].shape[1]], axis=0)
    return ret, index

def optShadow(imgs):
    ret = np.ndarray((imgs[0].shape[0], imgs[0].shape[1]))
    ret = (optMin(imgs)[0] + optMax(imgs)[0])//2
    return ret

# PRINT THOSE FRAMES AT RANDOM, IT'S SO PRETTY
def optDeltas(imgs):
    deltas = np.ndarray((imgs.shape[0], imgs[0].shape[0], imgs[0].shape[1]))
    for i, img in enumerate(imgs):
        deltas[i, 0:img.shape[0], 0:img.shape[1]] = (imgs[i, 0:img.shape[0], 0:img.shape[1]] - (optShadow(imgs)))
    return deltas

def optShadowTime(imgs, deltas, thresh):
    ret = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
    ret[0:ret.shape[0], 0:ret.shape[1]] = np.where(optMax(deltas[0:imgs[0].shape[0], 0:imgs[0].shape[1]])[0] - optMin(deltas[0:imgs[0].shape[0], 0:imgs[0].shape[1]])[0] > thresh, optMax(deltas[:imgs[0].shape[0], 0:imgs[0].shape[1]])[1], -1)
    return ret

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

'''imgs = np.array(imgs)
print(str(imgs.shape))
print(str(imgs[0].shape))
calib = np.array(calib)
shadows = optShadow(imgs)
deltas = optDeltas(imgs)
maxes = optMax(imgs)

window = plt.figure(figsize=(4,4))
print(str(shadows.shape) + " " + str(deltas.shape))
window.add_subplot(221)
plt.imshow(maxes[0], cmap='gray')
window.add_subplot(222)
plt.imshow(maxes[1], cmap='gray')
window.add_subplot(223)
plt.imshow(shadows, cmap='gray')
window.add_subplot(224)
plt.imshow(deltas[1], cmap='gray')
plt.show()'''

sys.setrecursionlimit(max(imgs[0].shape))

# EXTRACTING EDGES FOR POSTERIOR SPATIAL SHADOW MAPPING
cannyImgs = []
for img in imgs:
    cannyImgs.append(cv.Canny(img,85,255))
cannyImgs = np.array(cannyImgs)

# SPATIAL SHADOW LOCATION
# DETECT SHADOW BEHAVIOUR ON PLANE
spatial = spatialLocation(cannyImgs, ytop, ybottom)
print(spatial)

#shadowTime = np.frompyfunc(shadowTime, 4, 1)
#delta = np.frompyfunc(delta, 4, 1)

time = shadowTime(imgs, 560, 650, thresh)
print(time)

paintX(imgs[time], 560)
paintY(imgs[time], 650)

window = plt.figure(figsize=(8,4))
window.add_subplot(131)
plt.imshow(imgs[time], cmap='gray')

window.add_subplot(132)
plt.imshow(imgs[time+1], cmap='gray')

deltas = imageDelta(imgs, thresh, 255)

paintX(deltas[31], 123)
paintY(deltas[31], 505)

window.add_subplot(133)
plt.imshow(deltas[0], cmap='gray')

plt.show()

#plotPoints = optShadowTime(imgs, deltas, thresh)

#plt.imshow(plotPoints, cmap='gray')
#plt.show()

#window3d = plt.figure(figsize=(5, 4))
#axes = plt.axes(projection="3d")
#print(type(imgs[0].shape[0]))
#axes.scatter3D(range(imgs[0].shape[0]), range(imgs[0].shape[1]), shadowTime(imgs, range(imgs[0].shape[0]), range(imgs[0].shape[1]), thresh), c=plot, cmap='cividis')
#plt.show()


# DEBUG

#print(len(range(49)))
#print(len(plot))
#plt.scatter(range(50), plot[1:], color="black")
#plt.show()