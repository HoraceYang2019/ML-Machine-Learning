# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
13 features

1. CRIM: Per capita crime rate by town (人均犯罪率)
2. ZN: Proportion of residential land zoned for lots over 25,000 sq. ft (大面積民用土地比例)
3. INDUS: Proportion of non-retail business acres per town (非零售商土地比例)
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
5. NOX: Nitric oxide concentration (parts per 10 million) (一氧化氮濃度)
6. RM: Average number of rooms per dwelling (住宅平均房間數)
7. AGE: Proportion of owner-occupied units built prior to 1940 (老房率)
8. DIS: Weighted distances to five Boston employment centers (距市中心程度)
9. RAD: Index of accessibility to radial highways (至高速公路程度)
10. TAX: Full-value property tax rate per $10,000 (房屋稅)
11. PTRATIO: Pupil-teacher ratio by town (生師比)
12. B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
13. LSTAT: Percentage of lower status of the population (低收入比例)
14. MEDV: Median value of owner-occupied homes in $1000s

@author: hao
"""
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
# In[] Loading the Boston housing dataset
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(train_data[:,0], bins=30)
plt.show()

# In[] Normalizing the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# In[] build a model structure with parameters
from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# In[]  K-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
          # Prepares the validation data from partition #k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
          #Prepares the training data from all other partitions
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
    epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

np.mean(all_scores)
