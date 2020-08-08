# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:06:00 2019

@author: ASUS
"""
# In[1]:

import pandas as pd # 輸入
import numpy as np
insurance = pd.read_csv("data/insurance.csv") # 資料夾 data 下 insurance.csv

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    # to make this notebook's output identical at every run
    np.random.seed(42)  
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(insurance, 0.2)
insurance = train_set.drop("charges", axis=1)
insurance_labels = train_set["charges"].copy()

# ## 數值資料 3 個
# 
# * google pandas drop many columns
#     * https://stackoverflow.com/questions/28538536/deleting-multiple-columns-based-on-column-names-in-pandas

# In[2]:

insurance_num = insurance.drop(['sex', 'smoker', 'region'], axis=1)
insurance_num.head()

# In[3]:

from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
insurance.fillna(insurance.mode().iloc[0])
sex_ohe = encoder.fit_transform(insurance["sex"])
sex_ohe = np.hstack((1 - sex_ohe, sex_ohe))
sex_df = pd.DataFrame(sex_ohe, columns = encoder.classes_)
sex_df.head(10)

insurance.head(2)

# In[4]:

smoker_ohe = encoder.fit_transform(insurance["smoker"])
smoker_ohe = np.hstack((1 - smoker_ohe, smoker_ohe))
smoker_df = pd.DataFrame(smoker_ohe, columns = encoder.classes_)
smoker_df.head(2)

# In[5]:

region_ohe = encoder.fit_transform(insurance["region"])
region_df = pd.DataFrame(region_ohe, columns = encoder.classes_)
region_df.head(2)

# In[6]:
insurance_num = insurance_num.reset_index(drop=True)
insurance_df = pd.concat([insurance_num, sex_df, smoker_df, region_df], axis=1)
insurance_df.head()

# In[7]:

insurance_labels = insurance_labels.reset_index(drop=True)
insurance_labels.head() 

# # Betetr approach
# * https://github.com/pandas-dev/pandas/issues/12042

# In[8]:

cat_df = pd.concat([insurance["sex"], insurance["smoker"], insurance["region"]], axis=1)

pd.get_dummies(cat_df).head()

# In[9]:

all_data = pd.concat([insurance_df, insurance_labels], axis=1)
all_data.corr()
all_data.corr()["charges"].sort_values(ascending=False)

# In[10]:
# # 4.3 Select and Train a Model
# 
# * Linear Regression for training set
# 
#     * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#         * If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.


from sklearn.linear_model import LinearRegression
# Create a linear regressor instance
lr = LinearRegression(normalize=True)
# Train the model
lr.fit(insurance_df, insurance_labels)
print( "Score {:.4f}".format(lr.score(insurance_df, insurance_labels)) ) 

# infor 
# 
# * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#     * Returns the coefficient of determination R^2 of the prediction

print('y = %.3f '% lr.intercept_)
for i, c in enumerate(lr.coef_):
    print('%.3f '% c, insurance_df.columns.values[i])

# In[11]:

some_data = insurance_df.iloc[:4] # 4 個例子
some_labels = insurance_labels.iloc[:4]
print("Predictions:\t", lr.predict(some_data))
print("Labels:\t\t", list(some_labels))

# In[12] ## The 4th Sample
insurance_df.iloc[3]
insurance_labels.iloc[3]
lr.predict(insurance_df.iloc[3:5])

predicted_y = lr.intercept_

for i, c in enumerate(lr.coef_):
    predicted_y += c * insurance_df.iloc[3][i]
    
print('predicted y = %.3f '% predicted_y)

# In[13]: save the model and history 
import pickle
f = open('./outputs\\insurance.pckl', 'wb')
pickle.dump([train_set, test_set, insurance, insurance_num, cat_df, insurance_df,insurance_labels], f)
f.close()