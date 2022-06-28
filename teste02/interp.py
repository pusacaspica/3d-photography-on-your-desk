from ssl import PROTOCOL_TLSv1_2
import numpy as np
import matplotlib.pyplot as plt

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

def getClosestPoint(line1, line2):
    t = 0
    r1 = line1[0] + t * line1[1]
    r2 = line2[0] + t * line2[1]
    unitVectorNormal = np.cross(line1[1], np.transpose(line2[1]))/np.linalg.norm(np.cross(line1[1], np.transpose(line2[1])))
    distance = np.linalg.norm(np.dot(line2[0] - line1[0], np.cross(line1[1], np.transpose(line2[1])))//np.linalg.norm(np.cross(line1[1], np.transpose(line2[1]))))
    print(distance)
    print(unitVectorNormal)
    return unitVectorNormal, distance

pts = np.array([[4, 1, 2], [2, 5, 3], [5, 6, 2], [4, 2, 1]])
lines = np.array([[pts[1], pts[0]-pts[1]],  [pts[2], pts[3]-pts[2]]])

ptL1, ptL2, ptMid, dist = closestDistanceBetweenLines(pts[0], pts[1], pts[2], pts[3])
print(ptL1)
print(ptL2)
ptsClosest = np.concatenate(([ptL1], [ptL2]))
print(ptsClosest)
print(dist)
uvn, d = getClosestPoint(lines[0], lines[1])

ax = plt.axes(projection='3d')
ax.plot3D(pts[:2,0], pts[:2,1], pts[:2,2], 'green')
ax.plot3D(pts[2:,0], pts[2:,1], pts[2:,2], 'green')
ax.plot3D(ptsClosest[:, 0], ptsClosest[:, 1], ptsClosest[:, 2], 'red')
ax.scatter(ptMid[0], ptMid[1], ptMid[2])
plt.show()