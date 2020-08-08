#!/usr/bin/env python
# coding: utf-8
# In[1]:

import pandas as pd # 輸入
import numpy as np
insurance = pd.read_csv("data/insurance.csv") # 資料夾 data 下 insurance.csv

# In[2]
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
train_set.info()


# In[3]:

test_set.info()
train_set.head(2)
insurance = train_set.copy()
insurance.head(2)

# In[4]:  Single variable (單變量)

import matplotlib.pyplot as plt
insurance.charges.hist(bins = 3, figsize=(20,15))
plt.savefig("data//outputs")
plt.show()

# In[5]:
insurance.charges.hist(bins = 10)
plt.show()

# In[6]:
insurance.charges.min(), insurance.charges.max()
insurance.charges.describe()

# In[7]:Looking for Correlations (尋找相關性)
corr_matrix = insurance.corr()
corr_matrix 

corr_matrix["charges"].sort_values(ascending=False)
corr_matrix["charges"].sort_values()

# In[10]:

insurance.plot(kind="scatter", x="age", y="charges", alpha = 0.8)
plt.axis([18, 64, 1000, 65000])
plt.title('alpha = 0.8')
plt.savefig("data/age_vs_charges_scatterplot_8")
plt.show()

# In[11]:

insurance.plot(kind="scatter", x="age", y="charges", alpha= 0.2)
plt.axis([18, 64, 1000, 65000])
plt.title('alpha = 0.2')
plt.savefig("data/age_vs_charges_scatterplot_2")
plt.show()

# In[12]:

from pandas.plotting  import scatter_matrix

attributes = ["age", "bmi", "children" , "charges"]
scatter_matrix(insurance[attributes], figsize=(11, 8))
plt.savefig("data/scatter_matrix_plot")
plt.show()

# In[164]:Prepare the Data for Machine Learning Algorithms 
insurance = train_set.drop("charges", axis=1)
insurance_labels = train_set["charges"].copy()

insurance_labels.describe()
insurance.info()

insurance_labels.head(2)

# # 4.2.1 Dealing with missing Data

# In[172]:

import pandas as pd
insurance5 = pd.read_excel('data/insurance-5.xlsx') 
insurance5.head()

# * https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html

# In[173]:

insurance5.dropna()
insurance5.dropna().info()

# In[175]:‘all’ : If all values are NA, drop that row or column.

insurance5.dropna(how = 'all')
insurance5.bmi 
insurance5.bmi.mean()
insurance5.sex.mode()
insurance5.sex.mode()[0]

# google fillna multiple columns
# 
# * https://stackoverflow.com/questions/34913590/fillna-in-multiple-columns-in-place-in-python-pandas
# * https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.kind.html
# * https://www.w3schools.com/python/python_lambda.asp

# In[180]:

insurance5.head()
insurance5.fillna(insurance5.mode().iloc[0])
insurance5.head()
insurance5.fillna(insurance5.median()).fillna(insurance5.mode().iloc[0])

# ## (3) Random replace

# In[184]:

import pandas as pd # 輸入
insurance = pd.read_csv('data/insurance.csv')  # 資料夾 data 下 housing.csv
insurance.bmi.describe()
insurance.bmi.min(), insurance.bmi.max()
insurance.bmi.min().round(), insurance.bmi.max().round()


# In[187]:

import random
random.randrange(insurance.bmi.min().round(), insurance.bmi.max().round())
# google random element fillna
# 
# * https://stackoverflow.com/questions/47497466/python-fill-na-in-pandas-column-with-random-elements-from-a-list
# * https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html

# In[189]:

insurance.region.astype('category').values.categories
insurance.region.astype('category').values.categories[1]

region_cat = insurance.region.astype('category').values.categories

import numpy as np
np.random.choice(region_cat), np.random.choice(region_cat), np.random.choice(region_cat)

# In[193]: Managing categorical data (類別資料)
# * https://scikit-learn.org/stable/modules/preprocessing.html: The sklearn.preprocessing package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators. 

insurance_labels.head()
insurance.head()
insurance.region.astype('category').values.categories
insurance.region[0] 
insurance5.head()

# In[199]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
insurance_cat = insurance["region"]
insurance_cat_encoded = encoder.fit_transform(insurance_cat)
print(encoder.classes_)
print(insurance_cat_encoded)

# In[200]:

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
insurance_cat = insurance["region"]
insurance_cat_1hot = encoder.fit_transform(insurance_cat)
insurance_cat_1hot

# In[201]:
insurance_cat_1hot_df = pd.DataFrame(insurance_cat_1hot, columns = encoder.classes_)
insurance_cat_1hot_df.head()

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

# In[205]:Size of Data Frame

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

insurance6 = train_set.drop("charges", axis=0)
insurance6

# ## 數值資料 3 個
# 
# * google pandas drop many columns
#     * https://stackoverflow.com/questions/28538536/deleting-multiple-columns-based-on-column-names-in-pandas

# In[210]:

insurance_num = insurance.drop(['sex', 'smoker', 'region'], axis=1)
insurance_num.head()


# # Handling Text and Categorical Attributes
# 
# * 2 classes
#     * https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes

# In[211]:

insurance.head(10)


# In[212]:


from sklearn.preprocessing import LabelBinarizer
import numpy as np

encoder = LabelBinarizer()
sex_ohe = encoder.fit_transform(insurance["sex"])

# In[214]:


from sklearn.preprocessing import LabelBinarizer
import numpy as np

encoder = LabelBinarizer()
sex_ohe = encoder.fit_transform(insurance["sex"])
sex_ohe = np.hstack((1 - sex_ohe, sex_ohe))
sex_df = pd.DataFrame(sex_ohe, columns = encoder.classes_)
sex_df.head(10)

insurance.head(2)


# In[216]:


smoker_ohe = encoder.fit_transform(insurance["smoker"])
smoker_ohe = np.hstack((1 - smoker_ohe, smoker_ohe))
smoker_df = pd.DataFrame(smoker_ohe, columns = encoder.classes_)
smoker_df.head(2)


# In[217]:


insurance.head(2)


# In[218]:


region_ohe = encoder.fit_transform(insurance["region"])
region_df = pd.DataFrame(region_ohe, columns = encoder.classes_)
region_df.head(2)


# In[219]:


insurance.head()


# In[220]:


insurance_df = pd.concat([insurance_num, sex_df, smoker_df, region_df], axis=1)
insurance_df.head()


# In[221]:


insurance_labels = insurance_labels.reset_index(drop=True)
insurance_labels.head() 


# # Betetr approach
# 
# * https://github.com/pandas-dev/pandas/issues/12042

# In[222]:


cat_df = pd.concat([insurance["sex"], insurance["smoker"], insurance["region"]], axis=1)
cat_df.head()


# In[223]:


pd.get_dummies(cat_df).head()


# In[224]:


all_data = pd.concat([insurance_df, insurance_labels], axis=1)


# In[225]:


all_data.corr()


# In[226]:


all_data.corr()["charges"].sort_values(ascending=False)


# # 4.3 Select and Train a Model
# 
# * Linear Regression for training set
# 
#     * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#         * If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.

# In[227]:


from sklearn.linear_model import LinearRegression
# Create a linear regressor instance
lr = LinearRegression(normalize=True)
# Train the model
lr.fit(insurance_df, insurance_labels)


# infor 
# 
# * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#     * Returns the coefficient of determination R^2 of the prediction

# In[228]:


print( "Score {:.4f}".format(lr.score(insurance_df, insurance_labels)) ) 


# ### Linear Regression Equation

# In[229]:


print('y = %.3f '% lr.intercept_)
for i, c in enumerate(lr.coef_):
    print('%.3f '% c, insurance_df.columns.values[i])


# In[230]:


some_data = insurance_df.iloc[:4] # 4 個例子
some_labels = insurance_labels.iloc[:4]
print("Predictions:\t", lr.predict(some_data))
print("Labels:\t\t", list(some_labels))


# ## The 4th Sample

# In[231]:


insurance_df.iloc[3]


# In[232]:


insurance_labels.iloc[3]


# In[233]:


lr.predict(insurance_df.iloc[3:5])


# In[234]:


predicted_y = lr.intercept_

for i, c in enumerate(lr.coef_):
    predicted_y += c * insurance_df.iloc[3][i]
    
print('predicted y = %.3f '% predicted_y)


# In[235]:


insurance_all = pd.concat([insurance_df, insurance_labels], axis=1)
insurance_all.head()


# In[236]:


insurance_all[insurance_all.age == 52]


# In[237]:


insurance_all[(insurance_all.age == 52) & (insurance_all.female == 1) ]


# In[238]:


insurance_all[(insurance_all.age == 52) & (insurance_all.female == 1) & (insurance_all.children == 0) ]


# In[239]:


some_data = insurance_df.iloc[721:724]  
some_labels = insurance_labels.iloc[721:724]
print("Predictions:\t", lr.predict(some_data))
print("Labels:\t\t", list(some_labels))

lr.predict(insurance_df).min(), lr.predict(insurance_df).max()


# In[241]:


from sklearn.metrics import mean_squared_error
print(np.sqrt( mean_squared_error(insurance_labels, lr.predict(insurance_df) )))

insurance_labels.min(), insurance_labels.max()
lr.predict(insurance_df)
insurance_df.describe()

# In[245]: Multicollinearity (多重共線性) 
# * https://www.linkedin.com/pulse/super-simple-machine-learning-multiple-linear-regression-low/

cat_df.head(2)
cat_df.index

new_df = pd.DataFrame(index = cat_df.index)
for i in cat_df:
    new_df = new_df.join(pd.get_dummies(cat_df[i]).iloc[:, 1:])

new_df.head()


# In[248]:


insurance_new_df = pd.concat([insurance_num, new_df], axis=1)
insurance_new_df.head()


# In[249]:


lr2 = LinearRegression(normalize=True)
# Train the model
lr2.fit(insurance_new_df, insurance_labels)
print( "Score {:.4f}".format(lr2.score(insurance_new_df, insurance_labels)) ) 


# In[250]:


print('y = %.3f '% lr2.intercept_)
for i, c in enumerate(lr2.coef_):
    print('%.3f '% c, insurance_new_df.columns.values[i])


# ## 手動 StandardScaler , normalize= False 

# In[251]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaled_data2 = ss.fit_transform(insurance_new_df)

lr5 = LinearRegression(normalize= False)
lr5.fit(scaled_data2, insurance_labels)
print('y = %.3f '% lr5.intercept_)
for i, c in enumerate(lr5.coef_):
    print('%.3f '% c, insurance_new_df.columns.values[i])


# # 4.3.2 Evaluating model performance
# 
# * google significance linear regression python
#     * https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression/27928411

# In[252]:


from sklearn.preprocessing import StandardScaler
ss2 = StandardScaler()
scaled_data2 = ss2.fit_transform(insurance_new_df)


# In[253]:


import statsmodels.api as sm
#from scipy import stats

X2 = sm.add_constant(scaled_data2)
est = sm.OLS(insurance_labels, X2).fit()
print(est.summary())


# # 4.3.3 Backward selection (向後選擇) 

# In[254]:


insurance_df.head()


# In[255]:


insurance_back = insurance_df.drop(['female', 'male', 'no', 'northeast', 'northwest', 'southeast', 'southwest'], axis=1)
insurance_back.head()


# In[256]:


from sklearn.preprocessing import StandardScaler
ss6 = StandardScaler()
scaled_data6 = ss6.fit_transform(insurance_back)
lr6 = LinearRegression()
# Train the model
lr6.fit(scaled_data6, insurance_labels)
print( "Score {:.4f}".format(lr6.score(scaled_data6, insurance_labels)) ) 
print('y = %.3f '% lr6.intercept_)
for i, c in enumerate(lr6.coef_):
    print('%.3f '% c, insurance_back.columns.values[i])


# In[257]:


from sklearn.metrics import mean_squared_error
print(np.sqrt( mean_squared_error(insurance_labels, lr6.predict(scaled_data6) )))


# In[258]:


from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats

ss4 = StandardScaler()
scaled_data4 = ss.fit_transform(insurance_back)


X4 = sm.add_constant(scaled_data4)
est = sm.OLS(insurance_labels, X4)
est2 = est.fit()
print(est2.summary())


# In[259]:


est2.params


# In[260]:


est2.params[2]


# In[261]:


insurance_back.iloc[3]


# In[262]:


predicted_y = est2.params[0]

for i, c in enumerate(est2.params):
    if i == 0:
        predicted_y = est2.params[0]
    else: 
        predicted_y += c * insurance_back.iloc[3][i-1]
    
print('predicted y = %.3f '% predicted_y)


# In[263]:


est2.params[0] + insurance_back.iloc[3][0] * est2.params[1] + insurance_back.iloc[3][1] * est2.params[2] 


# # coefficients are different by using 2 methods
# 
# * solve the normal equation $\theta = (X^T X)^{-1} X^T y$ in lecture 5

# In[264]:


X = insurance_back.values
y = insurance_labels.values
scaled_data = ss.fit_transform(X)

X_b = np.c_[np.ones((insurance_back.shape[0], 1)), scaled_data]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
 
theta_best


# In[265]:


import numpy
numpy.linalg.eig(X_b.T.dot(X_b))


# In[266]:


X = insurance_new_df.values
y = insurance_labels.values
scaled_data = ss.fit_transform(X)

X_b2 = np.c_[np.ones((insurance_new_df.shape[0], 1)), scaled_data]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b2.T.dot(X_b2)).dot(X_b2.T).dot(y)
 
theta_best


# In[267]:


import numpy
numpy.linalg.eig(X_b2.T.dot(X_b2))


# # 4.3.4 Improving model performance 

# In[268]:


insurance_back.head()


# In[269]:


insurance_back['age2'] = insurance_back['age'] ** 2
insurance_back['bmi30_smoker'] =  (insurance_back['bmi'] > 30) * insurance_back['yes']
insurance_back.head(20)


# In[270]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
ss3 = StandardScaler()
scaled_data3 = ss3.fit_transform(insurance_back)

# Create a linear regressor instance
lr3 = LinearRegression()
# Train the model
lr3.fit(scaled_data3, insurance_labels)
print( "Score {:.4f}".format(lr3.score(scaled_data3, insurance_labels)) )


# In[271]:


print('y = %.3f '% lr3.intercept_)
for i, c in enumerate(lr3.coef_):
    print('%.3f '% c, insurance_back.columns.values[i])


# In[272]:


lr3.predict(insurance_back).min(), lr3.predict(insurance_back).max() 


# In[273]:


from sklearn.metrics import mean_squared_error
print(np.sqrt( mean_squared_error(insurance_labels, lr3.predict(scaled_data3) ))) 


# In[274]:


from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
# from scipy import stats

ss5 = StandardScaler()
scaled_data5 = ss5.fit_transform(insurance_back)

X5 = sm.add_constant(scaled_data5)
est = sm.OLS(insurance_labels, X5).fit()
print(est.summary())


# ## statistically nonsignificant (不顯著)

# In[275]:


insurance_back.head()


# In[276]:


insurance_back1 = insurance_back.drop(['age'], axis=1)
insurance_back1.head(2)


# In[277]:


ss6 = StandardScaler()
scaled_data6 = ss6.fit_transform(insurance_back1)


X6 = sm.add_constant(scaled_data6)
est = sm.OLS(insurance_labels, X6)
est2 = est.fit()
print(est2.summary())


# In[278]:


insurance_back2 = insurance_back1.drop(['bmi'], axis=1)
print(insurance_back2.head(2))
ss7 = StandardScaler()
scaled_data7= ss7.fit_transform(insurance_back2)


X7 = sm.add_constant(scaled_data7)
est = sm.OLS(insurance_labels, X7)
est2 = est.fit()
print(est2.summary())


# In[279]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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


# ## test set

# In[280]:


test_set.info()


# In[281]:


test_set.head(2)


# In[282]:


test_set['yes'] = (test_set["smoker"] == "yes").astype(int)
test_set.head(2)


# In[283]:


test_set['bmi30_smoker'] =  (test_set['bmi'] > 30) * test_set['yes']
test_set['age2'] = test_set['age'] ** 2
test_set.head(2)


# In[284]:


insurance_back2.head(2)


# In[285]:


test_set_df = pd.concat([test_set["children"], test_set["yes"], test_set["age2"], test_set["bmi30_smoker"]], axis=1)
test_set_df.head()


# In[286]:


test_set_df = test_set_df.reset_index(drop=True)
test_set_df.head()


# In[287]:


insurance_test_labels = test_set["charges"].copy()
insurance_test_labels = insurance_test_labels.reset_index(drop=True)
insurance_test_labels.head() 


# In[288]:


est2.predict(sm.add_constant(test_set_df))


# In[289]:


from sklearn.preprocessing import StandardScaler
ss8 = StandardScaler()
scaled_data8 = ss8.fit_transform(test_set_df)

print('Score %.3f' % lr8.score(scaled_data8, insurance_test_labels))


# In[290]:


from sklearn.metrics import mean_squared_error
# print(np.sqrt( mean_squared_error(insurance_labels, lr8.predict(insurance_df)) ))
print(np.sqrt( mean_squared_error(insurance_test_labels, lr8.predict(scaled_data8)) ))


# In[291]:


insurance_test_labels.min(), insurance_test_labels.max()


# In[292]:


lr8.predict(scaled_data8).min(), lr8.predict(scaled_data8).max()

