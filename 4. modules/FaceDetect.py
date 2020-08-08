# -*- coding: utf-8 -*-
"""
for windows
conda install -c menpo dlib 
pip install opencv-contrib-python

for linux
python -m pip install -U --force-reinstall dlib
python -m pip install -U --force-reinstall imutils

@author: hao
"""

import dlib
import cv2
import imutils

# In[]
image = cv2.imread('Photo/human.jpg')

# resize
image = imutils.resize(image, width=1280)

# call face detector
detector = dlib.get_frontal_face_detector()

# detect face
face_rects = detector(image, 0)

# extrract results
for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()

  # tag face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

cv2.imshow("Face Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()