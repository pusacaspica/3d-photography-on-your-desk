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

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# GET CLOSEST DISTANCE BETWEEN LINES EXTENDING TO INFINITY AND BEYOND
# FUNCTION BY Fnord ON STACK OVERFLOW: https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''
    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1) 
                
        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    midpoint = pA*0.5+pB*0.5
    print("midpoint: "+str(midpoint))
    
    return pA,pB,midpoint,np.linalg.norm(pA-pB)

# GET COLOR MAGNITUDE FROM PIXEL
def getColorMagnitude(pixel):
    return (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))//3

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
ObjTipPoints = np.array([
	np.array([397, 231.69, 70]),
	np.array([667.29, 295.08, 70]),
	np.array([405.93, -211.62, 70]),
	np.array([667.35, -220.34, 70])
])
ImgTipPoints = np.array([
    lampcalib[0][155-y, 883-x], lampcalib[0][506-y, 958-x],
    lampcalib[0][524-y, 26-x], lampcalib[0][161-y, 24-x]
])
ObjShadowPoints = np.array([
    np.array([471.29, 241.51, 0]),
	np.array([751.89, 307.75, 0]),
	np.array([479.86, -220.88, 0]),
	np.array([752.71, -229.41, 0])
])
ImgShadowPoints = np.array([
    lampcalib[0][156-y, 855-x], lampcalib[0][475-y, 921-x],
    lampcalib[0][490-y, 70-x], lampcalib[0][160-y, 58-x]
])

# LIGHT CALIBRATION
midpoints = []
_, _, midpoint1, _ = closestDistanceBetweenLines(ObjTipPoints[0], ObjShadowPoints[0], ObjTipPoints[1], ObjShadowPoints[1])
_, _, midpoint2, _ = closestDistanceBetweenLines(ObjTipPoints[0], ObjShadowPoints[0], ObjTipPoints[2], ObjShadowPoints[2])
_, _, midpoint3, _ = closestDistanceBetweenLines(ObjTipPoints[0], ObjShadowPoints[0], ObjTipPoints[3], ObjShadowPoints[3])
_, _, midpoint4, _ = closestDistanceBetweenLines(ObjTipPoints[1], ObjShadowPoints[1], ObjTipPoints[2], ObjShadowPoints[2])
_, _, midpoint5, _ = closestDistanceBetweenLines(ObjTipPoints[1], ObjShadowPoints[1], ObjTipPoints[3], ObjShadowPoints[3])
_, _, midpoint6, _ = closestDistanceBetweenLines(ObjTipPoints[2], ObjShadowPoints[2], ObjTipPoints[3], ObjShadowPoints[3])
midpoints.append(midpoint1)
midpoints.append(midpoint2)
midpoints.append(midpoint3)
midpoints.append(midpoint4)
midpoints.append(midpoint5)
midpoints.append(midpoint6)
midpoints = np.array(midpoints)
lightCenterPoint = np.array([np.sum(midpoints[:, 0])/len(midpoints), np.sum(midpoints[:, 1])/len(midpoints), np.sum(midpoints[:, 2])/len(midpoints)])
print(lightCenterPoint)

# DEBUG PLOTTING
ax = plt.axes(projection='3d')
ax.plot3D([ObjTipPoints[0, 0], ObjShadowPoints[0, 0]], [ObjTipPoints[0, 1], ObjShadowPoints[0, 1]], [ObjTipPoints[0, 2], ObjShadowPoints[0, 2]], 'green')
ax.plot3D([ObjTipPoints[1, 0], ObjShadowPoints[1, 0]], [ObjTipPoints[1, 1], ObjShadowPoints[1, 1]], [ObjTipPoints[1, 2], ObjShadowPoints[1, 2]], 'green')
ax.plot3D([ObjTipPoints[2, 0], ObjShadowPoints[2, 0]], [ObjTipPoints[2, 1], ObjShadowPoints[2, 1]], [ObjTipPoints[2, 2], ObjShadowPoints[2, 2]], 'green')
ax.plot3D([ObjTipPoints[3, 0], ObjShadowPoints[3, 0]], [ObjTipPoints[3, 1], ObjShadowPoints[3, 1]], [ObjTipPoints[3, 2], ObjShadowPoints[3, 2]], 'green')
ax.scatter(midpoint1[0], midpoint1[1], midpoint1[2], c= 'green')
ax.scatter(midpoint2[0], midpoint2[1], midpoint2[2], c= 'green')
ax.scatter(midpoint3[0], midpoint3[1], midpoint3[2], c= 'green')
ax.scatter(midpoint4[0], midpoint4[1], midpoint4[2], c= 'green')
ax.scatter(lightCenterPoint[0], lightCenterPoint[1], lightCenterPoint[2])
plt.show()


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