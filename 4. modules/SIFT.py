# -*- coding: utf-8 -*-
"""
Image matching 
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

for windows
pip install --user opencv-python==3.4.2.16
pip install --user opencv-contrib-python==3.4.2.16

for linux
sudo apt-get install python3-matplotlib

@author: hao
"""

import cv2
from matplotlib import pyplot as plt

cv2.__version__
# In[1]
img = cv2.imread('Photo/1.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None) # finds the keypoint in the images.
''' Each keypoint is a special structure which has many attributes like its (x,y) coordinates, 
    size of the meaningful neighbourhood, angle which specifies its orientation, 
    response that specifies strength of keypoints etc. '''

img=cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
''' draws the small circles on the locations of keypoints. 
If you pass a flag, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS to it, 
it will draw a circle with size of keypoint and it will even show its orientation.'''

cv2.imshow('img',img)
cv2.waitKey()

# In[]
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)

# In[3]
img1 = cv2.imread('Photo/7.jpg',cv2.IMREAD_GRAYSCALE)  # queryImage
img2 = cv2.imread('Photo/7.jpg',cv2.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

''' Match descriptors.
DMatch.distance - Distance between descriptors. The lower, the better it is.
DMatch.trainIdx - Index of the descriptor in train descriptors
DMatch.queryIdx - Index of the descriptor in query descriptors
DMatch.imgIdx - Index of the train image.
'''
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

plt.imshow(img3),plt.show()

# In[]
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2, good, None, flags=2)

plt.imshow(img3),plt.show()
