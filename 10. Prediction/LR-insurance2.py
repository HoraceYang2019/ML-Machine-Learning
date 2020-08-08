# -*- coding: utf-8 -*-
'''
Created on Thu Nov 21 14:06:00 2019

@author: ASUS
'''
import pandas as pd # 輸入
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np


# In[1]: 
f = open('./outputs\\insurance.pckl', 'rb')
[train_set, test_set, insurance, insurance_num, cat_df, insurance_df,insurance_labels] = pickle.load(f)
f.close()

# In[2]: Multicollinearity (多重共線性) 
# * https://www.linkedin.com/pulse/super-simple-machine-learning-multiple-linear-regression-low/

cat_df.head(2)
cat_df.index

new_df = pd.DataFrame(index = cat_df.index)
for i in cat_df:
    new_df = new_df.join(pd.get_dummies(cat_df[i]).iloc[:, 1:])

new_df.head()
insurance_new_df = pd.concat([insurance_num, new_df], axis=1)
insurance_new_df.head()

# In[3]:

lr2 = LinearRegression(normalize=True)
# Train the model
lr2.fit(insurance_new_df, insurance_labels)
print( "Score {:.4f}".format(lr2.score(insurance_new_df, insurance_labels)) ) 

print('y = %.3f '% lr2.intercept_)
for i, c in enumerate(lr2.coef_):
    print('%.3f '% c, insurance_new_df.columns.values[i])

# In[4]:

ss = StandardScaler()
scaled_data2 = ss.fit_transform(insurance_new_df)

lr5 = LinearRegression(normalize= False)
lr5.fit(scaled_data2, insurance_labels)
print('y = %.3f '% lr5.intercept_)
for i, c in enumerate(lr5.coef_):
    print('%.3f '% c, insurance_new_df.columns.values[i])

    
# In[5]:
# # 4.3.2 Evaluating model performance
# 
# * google significance linear regression python
# * https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression/27928411

ss2 = StandardScaler()
scaled_data2 = ss2.fit_transform(insurance_new_df)

# In[6]:

import statsmodels.api as sm
#from scipy import stats

X2 = sm.add_constant(scaled_data2)
est = sm.OLS(insurance_labels, X2).fit()
print(est.summary())


# # 4.3.3 Backward selection (向後選擇) 
insurance_df.head()
insurance_back = insurance_df.drop(['female', 'male', 'no', 'northeast', 'northwest', 'southeast', 'southwest'], axis=1)
insurance_back.head()

# In[7]:
ss6 = StandardScaler()
scaled_data6 = ss6.fit_transform(insurance_back)
lr6 = LinearRegression()
# Train the model
lr6.fit(scaled_data6, insurance_labels)
print( "Score {:.4f}".format(lr6.score(scaled_data6, insurance_labels)) ) 
print('y = %.3f '% lr6.intercept_)
for i, c in enumerate(lr6.coef_):
    print('%.3f '% c, insurance_back.columns.values[i])


print(np.sqrt( mean_squared_error(insurance_labels, lr6.predict(scaled_data6) )))

# In[8]:

import statsmodels.api as sm

ss4 = StandardScaler()
scaled_data4 = ss.fit_transform(insurance_back)

X4 = sm.add_constant(scaled_data4)
est = sm.OLS(insurance_labels, X4)
est2 = est.fit()
print(est2.summary())

# In[9]:

predicted_y = est2.params[0]
for i, c in enumerate(est2.params):
    if i == 0:
        predicted_y = est2.params[0]
    else: 
        predicted_y += c * insurance_back.iloc[3][i-1]
    
print('predicted y = %.3f '% predicted_y)

# In[10]:

est2.params[0] + insurance_back.iloc[3][0] * est2.params[1] + insurance_back.iloc[3][1] * est2.params[2] 

# # 4.3.4 Improving model performance 
# In[11]:
insurance_back['age2'] = insurance_back['age'] ** 2
insurance_back['bmi30_smoker'] =  (insurance_back['bmi'] > 30) * insurance_back['yes']
insurance_back.head(20)

# In[12]:
ss3 = StandardScaler()
scaled_data3 = ss3.fit_transform(insurance_back)

# Create a linear regressor instance
lr3 = LinearRegression()
# Train the model
lr3.fit(scaled_data3, insurance_labels)
print( "Score {:.4f}".format(lr3.score(scaled_data3, insurance_labels)) )

# In[13]:

print('y = %.3f '% lr3.intercept_)
for i, c in enumerate(lr3.coef_):
    print('%.3f '% c, insurance_back.columns.values[i])
lr3.predict(insurance_back).min(), lr3.predict(insurance_back).max() 

print(np.sqrt( mean_squared_error(insurance_labels, lr3.predict(scaled_data3) ))) 

# In[14]:

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
# from scipy import stats

ss5 = StandardScaler()
scaled_data5 = ss5.fit_transform(insurance_back)

X5 = sm.add_constant(scaled_data5)
est = sm.OLS(insurance_labels, X5).fit()
print(est.summary())

# In[15]:# ## statistically nonsignificant (不顯著)
insurance_back.head()
insurance_back1 = insurance_back.drop(['age'], axis=1)
insurance_back1.head(2)

ss6 = StandardScaler()
scaled_data6 = ss6.fit_transform(insurance_back1)

X6 = sm.add_constant(scaled_data6)
est = sm.OLS(insurance_labels, X6)
est2 = est.fit()
print(est2.summary())


# In[16]:

insurance_back2 = insurance_back1.drop(['bmi'], axis=1)
print(insurance_back2.head(2))
ss7 = StandardScaler()
scaled_data7= ss7.fit_transform(insurance_back2)

X7 = sm.add_constant(scaled_data7)
est = sm.OLS(insurance_labels, X7)
est2 = est.fit()
print(est2.summary())

# In[17]:
ss8 = StandardScaler()
scaled_data8 = ss3.fit_transform(insurance_back2)

# Create a linear regressor instance
lr8 = LinearRegression()
# Train the model
lr8.fit(scaled_data8, insurance_labels)
print( "Score {:.4f}".format(lr8.score(scaled_data8, insurance_labels)) )
print('y = %.3f '% lr8.intercept_)
for i, c in enumerate(lr8.coef_):
    print('%.3f '% c, insurance_back1.columns.values[i])

# In[18]:
test_set['bmi30_smoker'] =  (test_set['bmi'] > 30) * test_set['yes']

test_set_df = pd.concat([test_set["children"], test_set["yes"], test_set["age2"], test_set["bmi30_smoker"]], axis=1)
test_set_df.head()
test_set_df = test_set_df.reset_index(drop=True)
test_set_df.head()

test_set.info()

test_set.head(2)
test_set['yes'] = (test_set["smoker"] == "yes").astype(int)
test_set.head(2)


test_set['bmi30_smoker'] =  (test_set['bmi'] > 30) * test_set['yes']
test_set['age2'] = test_set['age'] ** 2
test_set.head(2)

insurance_back2.head(2)

test_set_df = pd.concat([test_set["children"], test_set["yes"], test_set["age2"], test_set["bmi30_smoker"]], axis=1)
test_set_df.head()

# In[19]:

test_set_df = test_set_df.reset_index(drop=True)
test_set_df.head()

# In[287]:

insurance_test_labels = test_set["charges"].copy()
insurance_test_labels = insurance_test_labels.reset_index(drop=True)
insurance_test_labels.head() 

est2.predict(sm.add_constant(test_set_df))

ss8 = StandardScaler()
scaled_data8 = ss8.fit_transform(test_set_df)

print('Score %.3f' % lr8.score(scaled_data8, insurance_test_labels))

# In[20]:

# print(np.sqrt( mean_squared_error(insurance_labels, lr8.predict(insurance_df)) ))
print(np.sqrt( mean_squared_error(insurance_test_labels, lr8.predict(scaled_data8)) ))
insurance_test_labels.min(), insurance_test_labels.max()
lr8.predict(scaled_data8).min(), lr8.predict(scaled_data8).max()
