# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 19:21:48 2018

@author: hao
"""

import numpy as np
A = np.mat("1 2; 3 2")
print("A: \n", A)
λ, U = np.linalg.eig(A) 

# In[] first eigenvector
i = 0
A.dot(U[:,i])
λ[i]*U[:,i]

# In[]
i = 1
A.dot(U[:,i])
λ[i]*U[:,i]

