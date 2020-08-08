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

# In[13]:Prepare the Data for Machine Learning Algorithms 
insurance = train_set.drop("charges", axis=1)
insurance_labels = train_set["charges"].copy()

insurance_labels.describe()
insurance.info()

insurance_labels.head(2)

# In[14]:Dealing with missing Data

import pandas as pd
insurance5 = pd.read_excel('data/insurance-5.xlsx') 
insurance5.head()

# In[173]:
# * https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html

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


# In[184]:
# ## (3) Random replace

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

insurance.head(2)

# In[218]:

region_ohe = encoder.fit_transform(insurance["region"])
region_df = pd.DataFrame(region_ohe, columns = encoder.classes_)
region_df.head(2)


insurance_df = pd.concat([insurance_num, sex_df, smoker_df, region_df], axis=1)
insurance_df.head()

# In[221]:
insurance_labels = insurance_labels.reset_index(drop=True)
insurance_labels.head() 

# In[222]:# Betetr approach
# https://github.com/pandas-dev/pandas/issues/12042

cat_df = pd.concat([insurance["sex"], insurance["smoker"], insurance["region"]], axis=1)
cat_df.head()

pd.get_dummies(cat_df).head()

all_data = pd.concat([insurance_df, insurance_labels], axis=1)
all_data.corr()

all_data.corr()["charges"].sort_values(ascending=False)