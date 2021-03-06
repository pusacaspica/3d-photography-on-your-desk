import scipy as sc
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def Normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).
    Input
    -----
    nd: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        
    Tr = np.linalg.inv(Tr)
    x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
    x = x[0:nd, :].T

    return Tr, x


def DLTcalib(nd, xyz, uv):
    '''
    Camera calibration by DLT using known object points and their image points.
    Input
    -----
    nd: dimensions of the object space, 3 here.
    xyz: coordinates in the object 3D space.
    uv: coordinates in the image 2D space.
    The coordinates (x,y,z and u,v) are given as columns and the different points as rows.
    There must be at least 6 calibration points for the 3D DLT.
    Output
    ------
     L: array of 11 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    if (nd != 3):
        raise ValueError('%dD DLT unsupported.' %(nd))
    
    # Converting all variables to numpy array
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)

    n = xyz.shape[0]

    # Validating the parameters:
    if uv.shape[0] != n:
        raise ValueError('Object (%d points) and image (%d points) have different number of points.' %(n, uv.shape[0]))

    if (xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' %(xyz.shape[1],nd,nd))

    if (n < 6):
        raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %(nd, 2*nd, n))
        
    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )

    # Convert A to array
    A = np.asarray(A) 

    # Find the 11 parameters:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]
    print(L)
    # Camera projection matrix
    H = L.reshape(3, nd + 1)
    print(H)

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
    print(H)
    H = H / H[-1, -1]
    print(H)
    L = H.flatten()
    print(L)

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = np.dot( H, np.concatenate( (xyz.T, np.ones((1, xyz.shape[0]))) ) ) 
    uv2 = uv2 / uv2[2, :] 
    # Mean distance:
    err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - uv)**2, 1)) ) 

    return L, err

def DLT():
    # Known 3D coordinates
    xyz = [[470.448 - 397.034, 241.187 + 229.608, 70.0],[397.034 - 397.034, 231.625 + 229.608, 0.0],
            [752.245 - 397.034, 307.869 + 229.608, 70.0],[667.416 - 397.034, 295.374 + 229.608, 0.0],
            [479.73 - 397.034, -220.556 + 229.608, 70.0],[406.089 - 397.034, -211.845 + 229.608, 0.0],
            [753.028 - 397.034, -229.608 + 229.608, 70.0], [667.474 - 397.034, -220.295 + 229.608, 0.0]]
    # Known pixel coordinates
    uv = [[499, 81],[534, 36],
            [168, 40],[168, 69],
            [489, 930],[515, 971],
            [163, 865],[163, 892]]

    nd = 3
    P, err = DLTcalib(nd, xyz, uv)
    print('Matrix')
    print(P)
    print('\nError')
    print(err)

    img = cv.imread("lamp_calibration_00.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.undistort(img, P, )
    plt.imshow(img, cmap='gray')

    

if __name__ == "__main__":
    DLT()