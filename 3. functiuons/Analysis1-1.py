# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:38:40 2019

@author: Horace Yang
"""

from pandas import read_excel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# In[0]
#-------------------------------------------------------------------------------------------------
expNo = 1 # specify what exp to check
file_name = './Source\\#545精加工段RawData示例v1.0.xlsx' # name of your excel file
#-------------------------------------------------------------------------------------------------
# In[1]
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    
    y=np.empty([data.shape[0],data.shape[1]]) 
    for i in range(0, data.shape[1]):
        y[:,i] = lfilter(b, a, data.iloc[:,i])
    y[np.isnan(y)] = 0  # fill na with zero
    return np.array(list(y[:,:]), dtype=np.float)  # return float matrix

# In[2]: specify the parameters by signal
# offset: signal offset, ex. current sesnor output 4-20 mA converted from 1V to 5 V
# scale: valid signal scale ranged from 1-5 V
# MaxRamge: actual detecting range of sensor, ex. current range from 0 -200A
spindle = {'sheet': 'Spindle Current', 'offset': 1.0, 'scale': 4.0, 'MaxRange': 200.0, 'cutoff': 120, 'unit': '(A)' }
xAxis = {'sheet': 'X axis Current', 'offset': 1.0, 'scale': 4.0, 'MaxRange': 50.0, 'cutoff': 120, 'unit': '(A)'}
zAxis = {'sheet': 'Z axis Current', 'offset': 1.0, 'scale': 4.0, 'MaxRange': 50.0, 'cutoff': 120, 'unit': '(A)'}
vib = {'sheet': 'Z-Vibration', 'offset': 0.0, 'scale': 5.0, 'MaxRange': 10.0, 'cutoff': 500,'unit': '(g)'}
param = [spindle, xAxis, zAxis, vib]
sampleRate = 2000

# In[3]:  Load raw data into dataframe
rTitle = [] # new list
rData = [] # new list

for i in range(0, len(param)):
    sheet_name = param[i]['sheet'] 
    df = read_excel(file_name, sheet_name = sheet_name)
    rTitle.append(sheet_name)  # 讀取感測來源
    rData.append(np.array(df.iloc[5:, expNo]))  # 複製實驗資料 from row 5:, col 1:
xData = pd.DataFrame(rData).T # transfer list to dataframe

# In[4]: 
# transform voltage levels to actual current levels 
order = 3 # filter order
nData = np.empty([xData.shape[0],xData.shape[1]]) 
fData = np.empty([xData.shape[0],xData.shape[1]]) 
for i in range(0, len(param)):
    offset = param[i]['offset']  
    scale = param[i]['scale']  
    MaxRange = param[i]['MaxRange']   # 實際量測範圍
    
    cutoff = param[i]['cutoff']  # desired cutoff frequency of the filter, Hz
    nData[:,i] = (xData.iloc[:,i]-offset) * MaxRange /scale
    b, a = butter_lowpass(cutoff, sampleRate, order=order)
    fData[:,i] = lfilter(b, a, nData[:,i])

# In[5]: show the data by Experiment

window = [00, 8000]  # 取樣率為2000，取樣範圍
colNo = [0, len(param)] 
alpha = 2
ylimit = [fData.mean()-alpha*fData.std(), fData.mean()+alpha*fData.std()]
xRange = range(window[0],window[1])
plt.suptitle('EXP'+str(expNo))

for i in range(colNo[0], colNo[1]): # plot specified data
    plt.subplot(colNo[1]-colNo[0], 1, i-colNo[0]+1);    
    
    plt.title(rTitle[i]);
    df=pd.DataFrame({'x': xRange,
                     'y0': nData[window[0]:window[1], i], # raw data
                     'y1': fData[window[0]:window[1], i]})     # filtered data

    plt.xlim(window[0], window[1]); 
    plt.xlabel('Sample Points(pts) in 2000 Hz')
    
    plt.ylabel(param[i]['unit'])
    
    plt.plot( 'x', 'y0', data=df, 
             marker='o', markerfacecolor='blue', markersize=2, color='skyblue', linewidth=2)
    plt.plot( 'x', 'y1', data=df, 
             marker='', color='olive', linewidth=2)
    plt.grid(True)

plt.show()

# In[]: AC motor power and torque
# http://pemclab.cn.nctu.edu.tw/PELIB/%E6%8A%80%E8%A1%93%E5%A0%B1%E5%91%8A/TR-001.%E9%9B%BB%E5%8B%95%E6%A9%9F%E6%8E%A7%E5%88%B6%E7%B0%A1%E4%BB%8B/html/
# Irms = 
# for spindle 
#CurrentRange = 200
#offset = 1.0
#IRMS_no =  # rms voltage (V)
#VRMS_no = 220  # rms voltage (V)
#zeta_m = 0.7 
#Pt = IRMS_no * VRMS_no * zeta_m
#
#IRMS_load =   # rms current (A)
#VRMS_load = 220 # rms voltage (V)
#Dm =  # workpiece diameter (mm) 
#f = 0.25  # feed (mm/rev)
#ap = [0, 0.1] # depth of cut (mm) 
#Kc = 800 # Mpa
#
#Pc = np.pi*Dm*f*ap*kc/(60*np.power(10,6)*zetz_m)
#
## T: total torque from 3-phase ac motor
#T =(3.0/2.0)* k * Im * Bm # Im: max phase current; Bm: max magnetic 
