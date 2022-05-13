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
    return imgs[t - 1][x][y]

# GET SPECIFIC t-NTH IMAGE
def getImg(imgs, t):
    return imgs[t]

# GET THE MAXIMAL INTENSITY VALUE,
# AND THE IMAGE IN WHICH THE RETURNED VALUE WAS FOUND
def getMaxMag(imgs):
    maxMag, maxIndex = 0, 0
    for indexes, img in enumerate(imgs[1:-2]):
        if(np.max(img.ravel()) > maxMag):
            maxMag = np.max(img)
            maxIndex = indexes
    return maxMag, maxIndex

# GET THE MINIMAL INTENSITY VALUE,
# AND THE IMAGE IN WHICH THE RETURNED VALUE WAS FOUND
def getMinMag(imgs):
    minMag, minIndex = 0, 0
    for indexes, img in enumerate(imgs[0:-2]):
        if(np.min(img.ravel()) < minMag):
            minMag = np.min(img)
            minIndex = indexes
    return minMag, minIndex

# GET THE AVERAGE SHADOW VALUE IN THE IMAGE
def getShadow(imgs):
    return (getMinMag(imgs) + getMaxMag(imgs))//2

# GET THE DIFFERENCE BETWEEN A SINGLE PIXEL 
# AND THE AVERAGE SHADOW VALUE
def getDelta(imgs, x, y, t):
    return getPx(imgs, x, y, t) - getShadow(imgs)

#def shadowLocation():

#def temporalLocation():

path = "./" # Should path be an input?
imgs, calib = [], []
ytop = int(input('enter ytop: '))
ybottom = int(input('enter ybottom: '))

# READING FILES IN PATH
for files in os.scandir(path):
    if files.name.rfind(".png"):
        if re.match('lamp_calibration_*', files.name):
            calib.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))
        else:
            imgs.append(cv.imread(files.name, cv.IMREAD_GRAYSCALE))

# EXTRACTING EDGES FOR POSTERIOR SHADOW/LOCATION MAPPING
cannyImgs = []
for img in imgs:
    cannyImgs.append(cv.Canny(img,100,200))

# DEBUG
window = plt.figure(figsize=(8,4))
window.add_subplot(121)
plt.imshow(imgs[0])

window.add_subplot(122)
plt.imshow(cannyImgs[0], cmap='gray')

plt.show()