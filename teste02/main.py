from email.mime import image
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

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def getQuaternionFromRotationMatrix(rmat):
    '''Converts opecv's rotation matrix got from cv.Rodrigues to quaternion array
       Function adapted from https://gist.github.com/shubh-agrawal/76754b9bfb0f4143819dbd146d15d4c8'''
    Q = np.zeros((1, 4)).ravel()
    print(Q)
    trace = np.trace(rmat)
    if(trace > 0):
        s = np.sqrt(trace + 1)
        Q[3] = s * 0.5
        s = 0.5/s
        Q[0] = (rmat[2, 1] - rmat[1, 2]) * s
        Q[1] = (rmat[0, 2] - rmat[2, 0]) * s
        Q[2] = (rmat[1, 0] - rmat[0, 1]) * s
    else:
        if rmat[0,0] < rmat[1,1]:
            if rmat[1,1] < rmat[2,2]:
                i = 2
            else:
                i = 1
        else:
            if rmat[0,0] < rmat[2,2]:
                i = 2
            else:
                i = 0
        j = (i + 1) % 3
        k = (i + 2) % 3

        s = np.sqrt(rmat[i, i] - rmat[j, j] - rmat[k, k] + 1)
        Q[i] = s * 0.5
        s = 0.5/s
        Q[3] = rmat[k,j] - rmat[j,k] * s
        Q[j] = rmat[j,i] + rmat[i,j] * s
        Q[k] = rmat[k,i] + rmat[i,k] * s
    return Q

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
    
    cross = np.cross(_A, _B)
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
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom
    t1 = detB/denom

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
    
    return pA,pB,midpoint,np.linalg.norm(pA-pB)

def getPlaneFromThreePoints(pt1, pt2, pt3, homogeneize=False):
    ''' Given three points defined by numpy.array triples (X, Y, Z)
        Return the matrix definition of the correspondent plane and the correspondent normal
    '''
    line1 = np.array([pt2[0], pt2[1], pt2[2]]) - np.array([pt1[0], pt1[1], pt1[2]])
    line2 = np.array([pt3[0], pt3[1], pt3[2]]) - np.array([pt1[0], pt1[1], pt1[2]])
    normal = np.cross(line1, line2)
    planeD = np.array([np.dot(normal, pt1)])
    #print("PlaneD: "+str(planeD))
    #if(planeD == 0):
    #    print("pt1: "+str(pt1)+"; pt2: "+str(pt2)+"; pt3: "+str(pt3))
    plane = np.concatenate((normal, -planeD))
    if(homogeneize == False):
        return plane, normal
    else:
        if(np.abs(planeD[0]) > 0):
            return np.array([plane/planeD[0], normal, planeD])
        else:
            return np.array([plane, normal])

# GET OBJECT POINT IN IMAGE POINT ONCE EXTRINSIC PARAMETERS ARE KNOWN
# HEAVILY BASED ON Milo's EXPLANATION AT https://stackoverflow.com/questions/14514357/converting-a-2d-image-point-to-a-3d-world-point
# AND https://stackoverflow.com/questions/32849373/how-do-i-obtain-the-camera-world-position-from-calibratecamera-results
def getObjectPointFromImagePoint(imagePoint, optMtx, rvecs, tvecs, tabletop = False, height = 7000):
    extrinsic = np.append(np.linalg.inv(rvecs), -tvecs).reshape(4, 3)
    img = np.dot(optMtx, imagePoint)
    #print("extrinsic: "+str(extrinsic))
    #print("imgpoint: "+str(imagePoint))
    result = np.dot(extrinsic , img)
    result /= result[3]
    #print("result:" + str(result))
    if (tabletop == True):
        return np.array(result[0:3]) * np.array([100, 100, 0])
    else:
        return np.array(result[0:3]) * np.array([100, 100, 0]) + np.array([0, 0, height])

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
    ret = np.where(optMax(imgs)[0] - optMin(imgs)[0] > 15, np.where(deltas[0] > thresh, deltas[1] + 1, -1), -1)
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

for imgcam in camcalib:
    ret, corners = cv.findChessboardCorners(imgcam, (7, 6), None)
    print(corners.ravel())
    centerImgOffset = np.array([imgs[0].shape[0], imgs[0].shape[1]])/2 - np.array([corners.ravel()[0], corners.ravel()[1]]) 
    print(centerImgOffset)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(imgcam, corners, (11,11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(imgcam, (7, 6), corners2, ret)
        plt.imshow(imgcam, cmap='gray')
        plt.show()

# print(objpoints)

corners = np.array(corners)
print(corners.shape)
corners = corners.reshape(corners.shape[0], corners.shape[2])
objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)
print(imgpoints.shape)
imgpoints = imgpoints.reshape(imgpoints.shape[0], imgpoints.shape[1],imgpoints.shape[3])
#print(corners.shape)
#print(objpoints.shape)
#print(imgpoints.shape)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, camcalib[0].shape[::-1], None, None)
rmat, tvec, _, _, _, _, _, _, _, _= cv.composeRT(rvecs[0], tvecs[0], rvecs[1], tvecs[1])
print("tvec: "+str(tvec))
print("rmat: "+str(rmat))
tmat = -rmat.T * tvecs
rmat, _ = cv.Rodrigues(rmat)
print("rmat: "+str(rmat))
invRmat = np.linalg.inv(rmat)
print("invRmat: "+ str(invRmat))
#invRmat = np.diagonal(np.linalg.inv(rvecs[0].ravel() * np.eye(3)))
invRvec, _ = cv.Rodrigues(invRmat)
print("invRvec: "+ str(invRvec))
if np.sqrt(rmat[1,0]**2 + rmat[0,0]**2) > 1e-6:
    camX, camY, camZ = camAngles = np.array([np.degrees(np.arctan2(rmat[2,1], rmat[2,2])), np.degrees(np.arctan2(-rmat[2, 0], np.sqrt(rmat[1,0]**2 + rmat[0,0]**2))), np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))])
else:
    camX, camY, camZ = camAngles = np.array([np.degrees(np.arctan2(-rmat[1,2], rmat[1,1])), np.degrees(np.arctan2(-rmat[2, 0], np.sqrt(rmat[1,0]**2 + rmat[0,0]**2))), np.degrees(0)])
print("angles: "+str(camAngles))
#print("invRmat: "+str(invRmat))

plotx = np.linspace(-512, 512)
ploty = np.linspace(-384, 384)
plotx, ploty = np.meshgrid(plotx, ploty)
#print(plotx.shape)
tb = (invRvec[2] * (plotx-tvecs[0][0]) + invRvec[0] * (ploty-tvecs[0][1])) / invRvec[1]#*(-tvecs[0][2])
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(plotx, ploty, tb - tvecs[0][2])
ax.scatter(tvecs[0][0], tvecs[0][1], tvecs[0][2], c='red')
plt.show()

h, w = camcalib[0].shape[:2]
optMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
x, y, w, h = roi
# print(roi)
print(lampcalib[0])
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
    #print(imgReversePoints.shape)
    imgReversePoints = np.ravel(imgReversePoints).reshape((imgReversePoints.shape[0], 2))
    error = cv.norm(imgpoints[i], imgReversePoints, cv.NORM_L2)/len(imgReversePoints)
    meanError += error
print("total error: {}".format(meanError/len(objpoints)))


# LIGHT CALIBRATION
# UNLESS I BOTHER TO IMPLEMENT A MORE ELEGANT SOLUTION
# LIGHT CALIBRATION COORDINATES WILL BE HARDCODED UNTIL THEN THEN
# HARDCODING CALIBRATION COORDINATES IS REALLY UNHEALTHY IN PYTHON
#print(calibImgs[0].shape)
ImgTipPoints = np.array([
    np.array([155,  883, 1]), 
    np.array([506 , 958, 1]),
    np.array([524 , 26 , 1]), 
    np.array([161 , 24 , 1])
])
print(ImgTipPoints)
ImgShadowPoints = np.array([
    np.array([156, 855, 1]), 
    np.array([475, 921, 1]),
    np.array([490, 70 , 1]), 
    np.array([160, 58 , 1])
])
'''print(ImgShadowPoints)
print(calibLampCalib.shape)
calibLampCalib[ImgTipPoints[0][0], ImgTipPoints[0][1]] = 255
calibLampCalib[ImgTipPoints[1][0], ImgTipPoints[1][1]] = 255
calibLampCalib[ImgTipPoints[2][0], ImgTipPoints[2][1]] = 255
calibLampCalib[ImgTipPoints[3][0], ImgTipPoints[3][1]] = 255
plt.imshow(calibLampCalib, cmap='gray')
plt.show()'''
# USING Vlad'S STRATEGY AS SEEN IN https://stackoverflow.com/questions/22389896/finding-the-real-world-coordinates-of-an-image-point
rvecs = rvecs[0].ravel()
tvecs = tvecs[0].ravel()
ObjTipPoints = []
ObjShadowPoints = []
for imgPoint in ImgTipPoints:
    objPoint = getObjectPointFromImagePoint(imgPoint, optMtx, rmat, tvecs, False, 7000) + np.array([tvecs[0], tvecs[1], 0]) #np.linalg.solve(np.transpose(rmat), (np.array([imgPoint[0] - w/2, h/2 - imgPoint[1], 1])-tvecs))
    ObjTipPoints.append(objPoint)
for imgPoint in ImgShadowPoints:
    objPoint = getObjectPointFromImagePoint(imgPoint, optMtx, rmat, tvecs, True) + np.array([tvecs[0], tvecs[1], 0])#np.linalg.solve(np.transpose(rmat), (np.array([imgPoint[0] - w/2, h/2 - imgPoint[1], 1])-tvecs))
    ObjShadowPoints.append(objPoint)
ObjTipPoints = np.array(ObjTipPoints)
ObjShadowPoints = np.array(ObjShadowPoints)
'''for imgPoint in ImgTipPoints:
    objPoint = ((imgPoint * rvecs) + tvecs)
    ObjTipPoints.append(objPoint)
ObjTipPoints = np.array(ObjTipPoints)
ObjShadowPoints = []
for imgPoint in ImgShadowPoints:
    objPoint = ((imgPoint* rvecs) + tvecs)
    ObjShadowPoints.append(objPoint)'''
'''ObjTipPoints = np.array([
	np.array([ 231.69 , 397   ,700]),
	np.array([ 295.08 , 667.29,700]),
	np.array([-211.62 , 405.93,700]),
	np.array([-220.34 , 667.35,700])
])
ObjShadowPoints = np.array([
    np.array([ 241.51 , 471.29, 0]),
	np.array([ 307.75 , 751.89, 0]),
	np.array([-220.88 , 479.86, 0]),
	np.array([ -230   , 752.71, 0])
])'''
ObjLines = np.array([
    np.array([ObjShadowPoints[0], ObjTipPoints[0] - ObjShadowPoints[0]]),
    np.array([ObjShadowPoints[1], ObjTipPoints[1] - ObjShadowPoints[1]]),
    np.array([ObjShadowPoints[2], ObjTipPoints[2] - ObjShadowPoints[2]]),
    np.array([ObjShadowPoints[3], ObjTipPoints[3] - ObjShadowPoints[3]])
])
print("obj shadow points: " + str(ObjShadowPoints))
print("obj tip points: " + str(ObjTipPoints))
print(invRmat)
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
ybottom = int(h * (ybottom/imgs[0].shape[0]))
ytop = int(h * (ytop/imgs[0].shape[0]))
spatial = []
for i in range(calibImgs.shape[0] - 1):
    spatial.append(cv.absdiff(calibImgs[i], calibImgs[i+1]))
    #plt.imshow(spatial[i], cmap='gray')
    #plt.show()

spatialEdges = spatialLocation(cannyImgs, ytop, ybottom)
#print(spatialEdges)
#plt.imshow(cannyImgs[0], cmap='gray')
#plt.show()
objShadowLine = []
shadowPlanes = []
for i, line in enumerate(spatialEdges):
    if(line[0] == -1 or line[1] == -1):
        continue
    ptTop = getObjectPointFromImagePoint(np.array([ytop, line[0], 1]), optMtx, rmat, tvecs, tabletop=True)
    ptBot = getObjectPointFromImagePoint(np.array([ybottom, line[1], 1]), optMtx, rmat, tvecs, tabletop=True)
    objShadowLine.append((ptTop, ptBot))
    #print(line)
    plane = np.array(getPlaneFromThreePoints(lightCenterPoint, ptTop, ptBot, True))
    #print(plane)
    shadowPlanes.append(plane)
    '''x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    x, y = np.meshgrid(x, y)
    z = (plane[0][3] - plane[1][0] * x - plane[1][1] * y) / plane[1][2]
    tb = np.array(invRmat[0] * (x) + invRmat[1] * (y)) #+ invRmat[2]*(tvecs[0][2]))
    plnPlot = plt.figure().gca(projection='3d')
    #plnPlot.plot(*zip(np.array([ytop, line[0], 1.0]), np.array([ybottom, line[1], 1.0]), lightCenterPoint))
    #plnPlot.plot3D([ytop, line[0]], [ybottom, line[1]])
    plnPlot.plot3D([ObjTipPoints[0, 1], ObjShadowPoints[0, 1]], [ObjTipPoints[0, 0], ObjShadowPoints[0, 0]], [ObjTipPoints[0, 2], ObjShadowPoints[0, 2]], 'green')
    plnPlot.plot3D([ObjTipPoints[1, 1], ObjShadowPoints[1, 1]], [ObjTipPoints[1, 0], ObjShadowPoints[1, 0]], [ObjTipPoints[1, 2], ObjShadowPoints[1, 2]], 'green')
    plnPlot.plot3D([ObjTipPoints[2, 1], ObjShadowPoints[2, 1]], [ObjTipPoints[2, 0], ObjShadowPoints[2, 0]], [ObjTipPoints[2, 2], ObjShadowPoints[2, 2]], 'green')
    plnPlot.plot3D([ObjTipPoints[3, 1], ObjShadowPoints[3, 1]], [ObjTipPoints[3, 0], ObjShadowPoints[3, 0]], [ObjTipPoints[3, 2], ObjShadowPoints[3, 2]], 'green')
    plnPlot.scatter(lightCenterPoint[1], lightCenterPoint[0], lightCenterPoint[2])
    plnPlot.plot_surface(x, y, z, cmap=plt.cm.coolwarm)
    plnPlot.plot_surface(x, y, tb)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()'''
objShadowLine = np.array(objShadowLine)
shadowPlanes = np.array(shadowPlanes)
print("shadow plane 0: " + str(shadowPlanes[0][0]))
sumPlanes = np.sum(spatial)
imgplane = sumPlanes - spatial[72] - spatial[73]
#plt.imshow(imgplane)
#plt.show()

# DEBUG PLOTTING
for i, plane in enumerate(shadowPlanes):
    ax = plt.figure().add_subplot(projection='3d')
    plotx = np.linspace(-160, 0)
    ploty = np.linspace(-160, 0)
    plotx, ploty = np.meshgrid(plotx, ploty)
    tb = (invRvec[2] * (plotx-tvecs[0]) + invRvec[0] * (ploty-tvecs[1])) * invRvec[1]#*(-tvecs[0][2])
    ax.plot_surface(plotx, ploty, tb , cmap=plt.cm.plasma)
    plotx = np.linspace(objShadowLine[i][1][1], objShadowLine[i][0][1])
    ploty = np.linspace(objShadowLine[i][1][0], objShadowLine[i][0][0])
    #ax.plot3D([objShadowLine[i][0][1], objShadowLine[i][1][1]],[objShadowLine[i][0][0], objShadowLine[i][1][0]], [objShadowLine[i][0][2], lightCenterPoint[2]], 'green')
    plotx, ploty = np.meshgrid(plotx, ploty)
    print("obj shadow line: "+ str(objShadowLine[i]))
    plotx, ploty = np.meshgrid(plotx, ploty)
    z = (-plane[0][3] - (plane[1][0]) * (plotx) - (plane[1][1]) * (ploty)) / (plane[1][2])
    #ax.plot_surface(ploty, plotx, z - tvecs[2], cmap=plt.cm.coolwarm)
    ax.scatter(tvecs[0], tvecs[1], tvecs[2], c = 'red')
    ax.plot3D([ObjTipPoints[0, 0], ObjShadowPoints[0, 0]],[ObjTipPoints[0, 1], ObjShadowPoints[0, 1]], [ObjTipPoints[0, 2], ObjShadowPoints[0, 2]], 'green')
    ax.plot3D([ObjTipPoints[1, 0], ObjShadowPoints[1, 0]],[ObjTipPoints[1, 1], ObjShadowPoints[1, 1]], [ObjTipPoints[1, 2], ObjShadowPoints[1, 2]], 'blue')
    ax.plot3D([ObjTipPoints[2, 0], ObjShadowPoints[2, 0]],[ObjTipPoints[2, 1], ObjShadowPoints[2, 1]], [ObjTipPoints[2, 2], ObjShadowPoints[2, 2]], 'red')
    ax.plot3D([ObjTipPoints[3, 0], ObjShadowPoints[3, 0]],[ObjTipPoints[3, 1], ObjShadowPoints[3, 1]], [ObjTipPoints[3, 2], ObjShadowPoints[3, 2]], 'orange')
    ax.scatter(midpoint1[0], midpoint1[1], midpoint1[2], c= 'green')
    ax.scatter(midpoint2[0], midpoint2[1], midpoint2[2], c= 'green')
    ax.scatter(midpoint3[0], midpoint3[1], midpoint3[2], c= 'green')
    ax.scatter(midpoint4[0], midpoint4[1], midpoint4[2], c= 'green')
    ax.scatter(lightCenterPoint[0], lightCenterPoint[1], lightCenterPoint[2], c='blue')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
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

print(shadowTime[268, 836])
plt.imshow(spatial[34], cmap='gray')
plt.show()
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
        delta = shadowTime[i, j]
        vertices.append(j%calibImgs[0].shape[1])
        vertices.append(shadowTime[i, j])
        vertices.append(i)
vertices = np.array(vertices, dtype=np.int32)
entry = {"vertices": vertices}
with open("mesh.json", "w") as write_file:
    json.dump(entry, write_file, cls=NumpyArrayEncoder)
#vertices += ']'

final = plt.axes(projection = '3d')
final.scatter(tvecs[0], tvecs[1], tvecs[2], c='red')
plt.text(zip(*tvecs)*1.1, s='camera', fontsize=12)
plt.show()

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