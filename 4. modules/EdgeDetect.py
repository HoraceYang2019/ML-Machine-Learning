# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 20:00:25 2018
https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
@author: hao
"""

import cv2
from matplotlib import pyplot as plt

# In[1]: Load and preprocess
img = cv2.imread('Photo/Windows-1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# remove noise
gray = cv2.GaussianBlur(gray,(3,3),0)

# In[]

laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=5)  # x
''' parameter1: image
    parameter2: cv2.CV_64F(64 bits float)  Depth of output image is passed -1 to get the result in np.uint8 type.
    parameter3: X gradient 0: original; 1: 1 order; 2: 2 order
    parameter4: Y gradient 0: original; 1: 1 order; 2: 2 order
    parameter5: kernel size default 3,  '''
    
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()    


# In[]
cv2.imwrite('Results/EdgeDetected.png', edges)
