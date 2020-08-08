# -*- coding: utf-8 -*-
'''
Created on Sun Jul 15 11:50:44 2018

@author: USER
'''

# In[1]: Data with operation 

height = 179.0
weight = 6870
bmi = weight / height ** 2
type(bmi)  # real numbers

day_of_week = 5
 #-----------
x = "body mass index"
y = 'this works too'
z= x+y

 #-----------
F1 = True
F2 = False
F1+F2

# In[2]: Index of Array and mtatrix
h1 = 1.73
h2 = 1.68
h3 = 1.71
h4 = 1.89

H1 = [h1, h2, h3, h4]

 # different data type 

H2 = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
H2[0]
H2[0][0]

 #--------- two dimensions
H3 = [["liz", 1.73],
        ["emma", 1.68],
        ["mom", 1.71],
        ["dad", 1.89]]

H3[0]
H3[0][0]
H3[0][0][0]
 # find elements
H2[-1] 
H2[-2]
H2[1:3]
H2[:4]
H2[5:]
# In[]
T = ('a','b','c','d',1,2,3)

T = ()
T = (1,)


# In[3] Operation of Array
H2[0:2] = ["lisa", 1.74]

 # 
del(H2[2])

tallest = max(H1)
shortest = min(H1)

round(16.68, 1) # 1 代表小數點右邊一位
round(16.68)  # 沒有第二個引數代表小數點右邊 0 位
help(round)
round(16.68,-1)  # -1 代表小數點左邊一位

H2.index("mom") # find the index 
H2.count(1.73) # count 
H2.count(1.71)

# In[4] String Operation
sister = "liz"
sister.capitalize()
sister.replace("z", "sa")

H2.append("me")
H2.append(1.79)

# In[5] package: numpy
import numpy
x=numpy.array([1, 2, 3])

import numpy as np
np.array([1, 2, 3])

from numpy import array
array([1, 2, 3])

W1 = [65.4, 59.2, 63.6, 88.4]
W1 / H1 ** 2

np_height = np.array(H1)
np_weight = np.array(W1)
bmi = np_weight / np_height ** 2

 #---- Different types: different behavior!
python_list = [1, 2, 3]
python_list + python_list

numpy_array = np.array([1, 2, 3])
numpy_array + numpy_array

 #---- Numpy Subsetting
bmi[1]
bmi > 21
bmi[bmi > 21]

# In[6] 2D Numpy Arrays
np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],
                  [65.4, 59.2, 63.6, 88.4, 68.7]])

np_2d.shape   # 2 rows, 5 columns
np_2d[0]
np_2d[1]
np_2d[:,1:3]
np_2d[0][2]
np_2d[0,2]

# In[7]:
height = np.round(np.random.normal(1.75, 0.20, 5000), 2)
weight = np.round(np.random.normal(60.32, 15, 5000), 2)
np_city = np.column_stack((height, weight))
np.mean(np_city[:,0])
np.median(np_city[:,0])
np.std(np_city[:,0])

# In[26]:Ranges 

np.arange(5) # np.arange(end): An array starting with 0 of increasing consecutive integers, stopping before end.
# 一個數組以0開始的連續整數增加，停止結束之前。
np.arange(3, 9) # np.arange(start, end): An array of consecutive increasing integers from start, stopping before end
# 從start，stoppin開始的連續遞增整數數組結束前
np.arange(3, 30, 5) 

# In[]
D = {'Name':'Jivin', 'Age': 6};

dict1 = D.copy()

r = b'ASCII string'
