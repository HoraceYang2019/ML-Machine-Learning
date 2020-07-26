# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:53:23 2018

@author: USER
"""
# In[1] File I/O
vehicles = ['scooter\n', 'bike\n', 'car\n']
f1 = open('vehicles.txt', 'a')
f1.writelines(vehicles)
f1.close()

f2 = open('vehicles.txt')
data = f2.readlines()

# In[2] Dataframe
import pandas as pd

df = pd.read_csv("brics.csv")
 # column access
df.country #Column access
df["country"]

 # delete index
df = pd.read_csv("brics.csv", index_col = 0)
 #Row access
df.loc["Brazil"] 

#Element access
df.loc["China","capital"] 
df["capital"].loc["China"]
df.loc["China"]["capital"]

 #Add Column
df["on_earth"] = [True, True, True, True, True] 
df["density"] = df["population"] / df["area"] * 1000000
 
df.to_csv('brices2.csv')

# In[]: read column data from csv file

import csv

fileName= 'test.csv'
data = ['4', 'Jaky', 'M']

data[1]='Yang'
# formal form for reading file
try:
    csvfile = open(fileName, newline='\n')
    for rowdata in csv.reader(csvfile):
         print(rowdata)
    
except:
    print ("fail to read")
    exit(-1)
finally:
    csvfile.close();          

    
# compact form for reading file
with open(fileName, 'r', newline='\n') as f1:
    for rowdata in csv.reader(f1):
        print(rowdata)
       
# write a csv file            
with open(fileName, 'a', newline='') as f2:        
    rowdata = csv.writer(f2, delimiter=',')
    rowdata.writerow(data)        
 
# In[]: read file in json
import json
from pprint import pprint

# Define data
data = {'CNC':
        {'Spindle':{'Current':[38.1, 38.2, 39.0],
                    'Vib':{'x':0.2, 'y':0.15}},
        'X Motor':2.5,
        'Z Motor':1.3}}

data["CNC"]['Spindle']['Current'][0]=3
#-------------------------------------------------------------------------
with open('test.json', 'w') as f1:
  json.dump(data, f1, ensure_ascii=False, indent=0 )
  
with open('test.json') as f2:
    rData = json.loads(f2.read())
pprint(rData)  