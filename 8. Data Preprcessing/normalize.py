# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:57:03 2019

@author: ASUS
"""


# In[202]: Data scaling and normalization

import numpy as np
x = np.array([[1, 2],[2, 6], [6, 1]])

# In[203]:
from sklearn.preprocessing import MinMaxScaler

ss = MinMaxScaler()
scaled_data = ss.fit_transform(x)
scaled_data


# In[204]:
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_data = ss.fit_transform(x)