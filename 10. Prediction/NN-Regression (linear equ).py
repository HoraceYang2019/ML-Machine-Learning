# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:35:02 2018
https://keras.io/
https://morvanzhou.github.io/tutorials/machine-learning/keras/2-1-regressor/
@author: ASUS
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras import models
from keras import layers
import matplotlib.pyplot as plt # 可视化模块

# In[1]: create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X*X + +0.7*X + 8 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()
offset= 150
X_train, Y_train = X[:offset], Y[:offset]     # train 前 150 data points
X_test, Y_test = X[offset:], Y[offset:]       # test 后 50 data points

# In[2]: build a model
model = models.Sequential()
model.add(layers.Dense(10, input_dim=1))

model.add(layers.Dense(1, activation='relu'))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd', metrics=['mae'])

model.summary()
# In[3]: train the model
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)
        
# In[4]: test the model
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=len(X_test))
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# In[5]: show the testing result 
# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()