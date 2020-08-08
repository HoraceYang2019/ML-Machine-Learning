# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:20:11 2017

@author: ASUS
"""

import pandas as pd
import glob

from sklearn.base import BaseEstimator, TransformerMixin
# Merge files into a  file
class DataMerger(BaseEstimator, TransformerMixin):
    def __init__(self, directory):
        self.fp = glob.glob(directory+'*.csv')
        
         # do nothing 
    def fit(self, X, y=None):
        return self
    
    # 轉換特徵
    def transform(self, X):
        tFile = X
        df_list = []
        for filename in sorted(self.fp):
            df = pd.read_csv(filename, skiprows=[0], header = None, encoding = 'big5') 
             # skip first row, no header
             
            df_list.append(df)
        full_df = pd.concat(df_list, axis = 0, ignore_index=True) # no index

        full_df.to_csv(tFile, index = False) # no index
        return(tFile)

# In[]
if __name__ == '__main__':
       mr = DataMerger(directory='Source/') # source data files
       s0 = mr.fit_transform('Target/MergedData.csv')  # merge source files 
