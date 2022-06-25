import sys
import re
import os
import cv2 as cv
from cv2 import projectPoints
from cv2 import CALIB_USE_INTRINSIC_GUESS
import json
import numpy
import numpy as np
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=sys.maxsize)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    top = int(ytop)
    bot = int(ybottom)
    for img in imgs:
        if(np.nonzero(img[top])[0].size < 2):
            topContour = -1
        else:
            topContour = np.nonzero(img[top])[0].ravel()[0]
        if(np.nonzero(img[bot])[0].size < 2):
            bottomContour = -1
        else:
            bottomContour = np.nonzero(img[bot])[0].ravel()[0]
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
# Fully working now
# Crucial if we're going to do this
def optMin(imgs):
    ret = index = np.ndarray((imgs[0].shape[0], imgs[0].shape[1]))
    ret = np.min(imgs, axis=0)
    index = np.argmin(imgs, axis=0)
    return ret, index

def optMax(imgs):
    ret = index = np.ndarray((imgs[0].shape[0], imgs[0].shape[1]))
    ret = np.max(imgs, axis = 0)
    index = np.argmax(imgs, axis=0)
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
    return optMax(deltas)

def optShadowTime(imgs, deltas, thresh):
    ret = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
    ret = np.where(optMax(imgs)[0] - optMin(imgs)[0] > 30, np.where(deltas[0] > thresh, deltas[1] + 1, -1), -1)
    return ret

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

# EXTENDING RECURSION LIMIT BECAUSE WE CAN
sys.setrecursionlimit(imgs[0].shape[0])

# CAMERA CALIBRATION
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints =[] 
imgpoints = []

'''imgPoints_camera = [np.float32([[266, 182], [330, 182], [395, 181], [459, 180], [524, 180], [589, 180], [654, 179],
                   [259, 224], [324, 224], [390, 224], [458, 224], [524, 223], [591, 223], [659, 223],
                   [249, 271], [317, 270], [387, 270], [456, 270], [525, 269], [593, 269], [663, 269]])]
objPoints_camera = [np.float32([[0,0,0],[0,10,0],[0,20,0],[0,30,0],[0,40,0],[0,50,0],[0,60,0],
                   [10,0,0],[10,10,0],[10,20,0],[10,30,0],[10,40,0],[10,50,0],[10,60,0],
                   [20,0,0],[20,10,0],[20,20,0],[20,30,0],[20,40,0],[20,50,0],[20,60,0]])]'''

ret, corners = cv.findChessboardCorners(camcalib[0], (7, 6), None)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(camcalib[0], corners, (11,11), (-1, -1), criteria)
    imgpoints.append(corners2)
    '''cv.drawChessboardCorners(camcalib[0], (7, 6), corners2, ret)
    plt.imshow(camcalib[0], cmap='gray')
    plt.show()'''

corners = np.array(corners)
corners = corners.reshape(corners.shape[0], corners.shape[2])
objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)
imgpoints = imgpoints.reshape(imgpoints.shape[2], imgpoints.shape[1],imgpoints.shape[3])
print(corners.shape)
print(objpoints.shape)
print(imgpoints.shape)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, camcalib[0].shape[::-1], None, None)
print(rvecs)
print(tvecs)

plotx = ploty = np.linspace(-100, 100, 100)
plotx, ploty = np.meshgrid(plotx, ploty)
eq = rvecs[0][0] * (plotx-tvecs[0][0]) + rvecs[0][1] * (ploty-tvecs[0][1]) + rvecs[0][2]*(-tvecs[0][2])
fg = plt.figure()
ax = fg.gca(projection='3d')
ax.plot_surface(plotx, ploty, eq)
plt.show()


h, w = camcalib[0].shape[:2]
optMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
x, y, w, h = roi
print(roi)
lampcalib[0] = cv.undistort(lampcalib[0], mtx, dist, None, optMtx)

# UNDISTORTION
# I'M FOLLOWING A TUTORIAL AND WANTING TO SEE WHERE IT LEADS
for i, img in enumerate(imgs):
    calibrated = cv.undistort(img, mtx, dist, None, optMtx)
    calibImgs.append(calibrated[y:y+h, x:x+w])
calibImgs = np.array(calibImgs)
cv.imwrite("rawCalibratedImage.png", lampcalib[0])
calibLampCalib = lampcalib[0][y:y+h, x:x+w]
cv.imwrite("calibratedImage.png", calibLampCalib)

# REPROJECTION IN ORDER TO BE SURE THIS CALIBRATION IS WORKING
meanError = 0
for i in range(len(objpoints)):
    imgReversePoints,_ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    # print(imgReversePoints)
    imgpoints[i] = np.float32(imgpoints[i])
    print(imgReversePoints.shape)
    imgReversePoints = np.ravel(imgReversePoints).reshape((imgReversePoints.shape[0], 2))
    error = cv.norm(imgpoints[i], imgReversePoints, cv.NORM_L2)/len(imgReversePoints)
    meanError += error
print("total error: {}".format(meanError/len(objpoints)))

# LIGHT CALIBRATION
# UNLESS I BOTHER TO IMPLEMENT A MORE ELEGANT SOLUTION
# LIGHT CALIBRATION COORDINATES WILL BE HARDCODED UNTIL THEN THEN
# HARDCODING CALIBRATION COORDINATES IS REALLY UNHEALTHY IN PYTHON
imgPoints = [np.float32([[499, 81],[534, 36],
            [168, 40],[168, 69],
            [489, 930],[515, 971],
            [163, 865],[163, 892]])]
objPoints = [np.float32([[470.448 - 397.034, 241.187 + 229.608, 70.0],[397.034 - 397.034, 231.625 + 229.608, 0.0],
            [752.245 - 397.034, 307.869 + 229.608, 70.0],[667.416 - 397.034, 295.374 + 229.608, 0.0],
            [479.73 - 397.034, -220.556 + 229.608, 70.0],[406.089 - 397.034, -211.845 + 229.608, 0.0],
            [753.028 - 397.034, -229.608 + 229.608, 70.0], [667.474 - 397.034, -220.295 + 229.608, 0.0]])]
print(objPoints[0].size)
lightLines = []
# TO DO: INTERCEPT ALL THESE POINTS
# REAL LIFE POINTS ARE THE ONES WHO MATTER, MUST EXTRACT VALUES FROM imgPoints
# 1. CONVERT imgPoints TO OBJECT POINTS (multiply with camera matrix mtx?)
# 2. ISOLATE THE VECTOR OF THE LINE as seen on http://ramanujan.math.trinity.edu/rdaileda/teach/f20/m2321/lectures/lecture5_slides.pdf
# 3. USE LEAST SQUARES TO DETECT INTERSECTION


# SHADOW MAPPING

# EXTRACTING EDGES FOR POSTERIOR SPATIAL SHADOW MAPPING
# CALIBRATEDIMGS CAN BE CHANGED TO IMGS TO WORK WITH UNCALIBRATED IMAGES
# BUT FOR THE LOVE OF G-D IF YOU DO THIS YOU NEED TO CHANGE DISTANCES IN SPATIAL LOCATION
cannyImgs = []
for img in calibImgs:
    cannyImgs.append(cv.Canny(img,85,255))
cannyImgs = np.array(cannyImgs)

# SPATIAL SHADOW LOCATION
# DETECT SHADOW BEHAVIOUR ON PLANE
# IF USING UNCALIBRATED IMAGES FOR CANNY, USE ytop AND ybottom
# IF USING CALIBRATED IMAGES, USE int(h * (ytop/imgs[0].shape[0])) AND int(h * (ybottom/imgs[0].shape[0]))
spatial = spatialLocation(cannyImgs, int(h * (ytop/imgs[0].shape[0])), int(h * (ybottom/imgs[0].shape[0])))
print(spatial)
plt.imshow(cannyImgs[0], cmap='gray')
plt.show()
# NOT DONE YET
# SHADOW PLANES MUST BE ESTIMATED
# CAN ONLY BE DONE WHEN LIGHT CALIBRATION IS DONE

# EXTRACTING SHADOW AND DELTA FOR DEBUGGING PURPOSES
# CALIBRATEDIMGS CAN BE CHANGED TO IMGS TO WORK WITH UNCALIBRATED IMAGES
mins = optMin(calibImgs)[0]
maxes = optMax(calibImgs)[0]
shadows = optShadow(calibImgs)
deltas = optDeltas(calibImgs)

# TEMPORAL SHADOW LOCATION
# DETECT TIMESLICE IN WHICH FRAME IS TOUCHED BY MOVING SHADOW
shadowTime = optShadowTime(calibImgs, deltas, thresh)

# SHOW IMAGES
window = plt.figure(figsize=(4,4))
# print(str(shadows.shape) + " " + str(deltas[0].shape))
window.add_subplot(221)
plt.imshow(shadows, cmap='gray')
window.add_subplot(222)
plt.imshow(deltas[0], cmap='gray')
window.add_subplot(223)
plt.imshow(maxes, cmap='gray')
window.add_subplot(224)
plt.imshow(shadowTime, cmap='gray')
plt.show()

#vertices = '"vertices" : ['
vertices = []
for i, times in enumerate(shadowTime):
    #if (i > 0):
    #    vertices += ", "
    for j, time in enumerate(times):
        vertices.append(j%calibImgs[0].shape[1])
        vertices.append(shadowTime[i, j])
        vertices.append(i)
vertices = np.array(vertices, dtype=np.int32)
entry = {"vertices": vertices}
with open("mesh.json", "w") as write_file:
    json.dump(entry, write_file, cls=NumpyArrayEncoder)
#vertices += ']'

'''window3d = plt.figure(figsize=(7, 7))
axes = plt.axes(projection="3d")
axes.surface3D(imgs[0,:,0]*imgs.shape[2], imgs[0,:,0]*imgs.shape[1], shadowTime)
plt.show()'''

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