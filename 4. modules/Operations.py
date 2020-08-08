# -*- coding: utf-8 -*-
"""
pip install opencv-contrib-python
check opencv version: 
    cv2.__version__
    
https://docs.opencv.org/3.4.0/da/d6e/tutorial_py_geometric_transformations.html
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#install-opencv-python-in-windows
print (cv2.__version__)
# In[1]: original image
image = cv2.imread("Photo/0.jpg")
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

color = ('b','g','r')
for i, col in enumerate(color):
  histr = cv2.calcHist([image],[i],None,[256],[0, 256])
  plt.plot(histr, color = col)
  plt.xlim([0, 256])
plt.show()

# In[2]: resize
rows, cols = image.shape[:2]
resize = cv2.resize(image, (int(0.2*cols), int(0.2*rows)), interpolation = cv2.INTER_CUBIC)
cv2.imshow('Resize', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[3]: Convert RGB to Gray/HSV
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
 # cv2.calcHist(image, channel, mask, bins, pixel range)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
gray_r = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
hist_r = cv2.calcHist([gray_r], [0], None, [256], [0, 256])

plt.plot(hist, 'b')
plt.plot(hist_r, 'r')
plt.show()
cv2.waitKey(0)

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
cv2.waitKey(0)

# Convert HSV to RGB
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("BGR", bgr)
cv2.waitKey(0)

cv2.destroyAllWindows()

# In[4]: Mask Image 
mask = np.zeros(gray.shape, np.uint8)
mask[60:130, 50:180] = 255

# calculate image with mask
masked_gray = cv2.bitwise_and(gray, gray, mask = mask)

# historgram of orignial image
hist_full = cv2.calcHist([image], [0], None, [256], [0, 256])

# historgram of masked image
hist_mask = cv2.calcHist([image], [0], mask, [256], [0, 256])

# show result
plt.subplot(221), plt.imshow(gray, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_gray, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()

# In[5]: Shift/Rotate

def shift(image, x, y):
    # define shift matrix
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

shifted = shift(image, 50, 30)
cv2.imshow("Shifted Down", shifted)
cv2.waitKey(0)

angle = 30
rotated = rotate(image, angle)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)

# In[]
n,m=gray.shape
y0=np.full((1,m),255)
grayy = np.vstack((gray[1:,],y0))
dy=grayy-gray
