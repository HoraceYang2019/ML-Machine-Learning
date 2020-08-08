# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 07:46:05 2018
https://docs.opencv.org/3.4.0/d5/daf/tutorial_py_histogram_equalization.html
https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=equalizehist
@author: hao
"""
import numpy
import cv2
import matplotlib.pyplot as plt

# In[]
image = cv2.imread('Photo/img.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculates a histogram of a set of arrays.
hist_g = cv2.calcHist([gray], [0], None, [256], [0, 256]) 

#cv2.imshow("Gray", gray)
# Equalizes the histogram of a grayscale image.
eq = cv2.equalizeHist(gray) 


hist_e = cv2.calcHist([eq], [0], None, [256], [0, 256])

cv2.imshow("Histogram Equalization", numpy.hstack([gray, eq]))
cv2.imwrite("mountain_eq.jpg", numpy.hstack([gray, eq]))

plt.plot(hist_g, 'b'); plt.xlim([0, 256])
plt.plot(hist_e, 'r'); plt.xlim([0, 256])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
