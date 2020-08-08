# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 05:43:49 2019
https://blog.csdn.net/legalhighhigh/article/details/81409551
https://ithelp.ithome.com.tw/articles/10191725
@author: hao
"""

import tensorflow as tf
import numpy as np

def softmax(x):
    sum_raw = np.sum(np.exp(x),axis=-1)
    x1 = np.ones(np.shape(x))
    for i in range(np.shape(x)[0]):
        x1[i] = np.exp(x[i])/sum_raw[i]
    return x1

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

# In[1]: Binary Classes
# 5 samples with 2 classes, each sample with one class
wx = np.array([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]])
y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]])

y_head = sigmoid(wx)    
E0 = -y_head*np.log(y_head)-(1-y_head)*np.log(1-y_head) # binary cross entropy
E1 = -y*np.log(y_head)-(1-y)*np.log(1-y_head) # binary cross entropy
print(E1)

sess = tf.Session()
y = np.array(y).astype(np.float64)
E2 = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=wx))

print(E2)

# In[2]: categorical crossentropy
# 5 samples with 3 classes, each sample with multi-class
wx = np.array([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]])
y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,0]])

y_head = softmax(wx)

E1 = -np.sum(y*np.log(y_head),-1) # caltegorical cross entropy
print(E1)

sess =tf.Session()
y = np.array(y).astype(np.float64) 
E2 = sess.run(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=wx))
print(E2)


