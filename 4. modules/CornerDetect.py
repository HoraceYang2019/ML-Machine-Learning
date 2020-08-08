# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:40:27 2018
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornerharris
@author: hao
"""

import cv2
import numpy as np

# In[]
filename = 'Photo/dog.jpg' #blocks.jpg, toy.jpg
img = cv2.imread(filename)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)

'''
cv2.cornerHarris(img, blocksize, ksize, k)
img - Input image, it should be grayscale and float32 type.
blockSize - It is the size of neighbourhood considered for corner detection
ksize - Aperture parameter of Sobel derivative used.
k - Harris detector free parameter in the equation.
'''
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255] 

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    
# In[]
    
ret, dst = cv2.threshold(dst, 0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv2.imwrite('Results\CornerDetected.png',img)