# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:49:09 2018

@author: USER
"""

# In[1] logic comparison

2 < 3
2 == 3

x = 2
y = 3
x < y
x == y

# In[2] Conditional Statements
z = 4
if z % 2 == 0:
    print("z is even")

if z % 2 == 0 :
    print("checking " + str(z))
    print("z is even")
  
    #----- 
z = 5
if z % 2 == 0 :
    print("checking " + str(z))
    print("z is even")

z = 5
if z % 2 == 0 :
    print("z is even")
else :
    print("z is odd") 
    

    #----- 
z = 3
if z % 2 == 0 :
    print("z is divisible by 2")
elif z % 3 == 0 :
    print("z is divisible by 3")
else :
    print("z is neither divisible by 2 nor by 3")

    #----- 
z = 4
if z % 2 == 0 or z %3 == 0:
    if z % 3 == 0 :
        print("z is divisible by 3")
    if z % 2 == 0:
        print("z is divisible by 2")
else :
    print("z is neither divisible by 2 nor by 3")
 