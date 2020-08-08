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
sensor = 0  # 0: spindle, 1: x axis, 2: z axis, 3: vibration
file_name = './Source\\#545精加工段RawData示例v1.0' # name of your excel file
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
    
    y=np.empty([data.shape[0],data.shape[1]]) # for 2-D array 
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
vib = {'sheet': 'Z-Vibration', 'offset': 0.0, 'scale': 5.0, 'MaxRange': 10.0, 'cutoff': 250,'unit': '(g)'}
#Exp = {'1': [1500, 4000], '2':[], '3':[2500, 3500]}
param = [spindle, xAxis, zAxis, vib]
sampleRate = 2000
sheet_name = param[sensor]['sheet'] 

# In[3]:  Load raw data into dataframe

df = read_excel(file_name+'.xlsx', sheet_name = sheet_name)
df.fillna(0)
df.head()
rTitle = df.iloc[0,1:]  # 讀取實驗編號
rData = df.iloc[5:, 1:]  # 複製實驗資料 from row 5:, col 1:

# In[4]: 
# transform voltage levels to actual current levels 
offset = param[sensor]['offset']  
scale = param[sensor]['scale']  
MaxRange = param[sensor]['MaxRange']   # 實際量測範圍
unit = param[sensor]['unit']
nData = (rData-offset) * MaxRange /scale # transfer to actual value

# In[4.1]: filtered data
order = 3
cutoff = param[sensor]['cutoff']  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
fData = butter_lowpass_filter(nData, cutoff, sampleRate, order)

# In[5]: show the data by sensor

window = [0, 8000]  # 取樣率為2000，取樣範圍
colNo = [0, 13] # 
alpha = 2
ylimit = [fData.mean()-alpha*fData.std(), fData.mean()+alpha*fData.std()]
xRange = range(window[0],window[1])

fig = 0
maxFig = 5
expFeatures = np.empty([fData.shape[1], 1])
for i in range(colNo[0], colNo[1]): # plot specified data
    if i % maxFig == 0:
        fig += 1
        plt.figure(file_name+ '- Fig ' +str(fig))
    plt.subplot(maxFig, 1, (i%maxFig)+1);    
    plt.suptitle(sheet_name)
    plt.title(rTitle[i]);
    pdf=pd.DataFrame({'x': xRange,
                     'y0': nData.iloc[window[0]:window[1], i], # raw data
                     'y1': fData[window[0]:window[1], i]})     # filtered data
    expFeatures[i] = pdf['y1'].mean()
    plt.xlim(window[0], window[1]); 
    plt.xlabel('Sample Points(pts) in 2000 Hz')
    
    plt.ylim(ylimit[0], ylimit[1]); 
    plt.ylabel(unit)
    
    plt.plot( 'x', 'y0', data=pdf, 
             marker='o', markerfacecolor='blue', markersize=2, color='skyblue', linewidth=2)
    plt.plot( 'x', 'y1', data=pdf, 
             marker='', color='red', linewidth=2)
    plt.grid(True)

plt.show()
expFeatures 

# In[]
x = pd.DataFrame(data=expFeatures.T, columns=df.iloc[1,1:])
y = pd.DataFrame(data=No545.T, columns=df.iloc[1,1:])
x=x.append(y,ignore_index=True)
x.rename(index={0: "No544", 1: "No545"})
export_csv = x.to_csv ('expFeatures.csv', header=True) 

# In[]: AC motor power and torque
# http://pemclab.cn.nctu.edu.tw/PELIB/%E6%8A%80%E8%A1%93%E5%A0%B1%E5%91%8A/TR-001.%E9%9B%BB%E5%8B%95%E6%A9%9F%E6%8E%A7%E5%88%B6%E7%B0%A1%E4%BB%8B/html/

IRMS_no = fData[window[0]:window[1], 0:2].mean() # rms voltage (V)
VRMS_no = 220  # rms voltage (V)
zeta_m = 0.7 
Pt = IRMS_no * VRMS_no * zeta_m 
#
IRMS_load = fData[window[0]:window[1], 3].mean()  # rms current (A)
VRMS_load = 220 # rms voltage (V)

#Dm = 30 # workpiece diameter (mm) 
#f = 0.25  # feed (mm/rev)
#ap = 0.1 # depth of cut (mm) 
#Kc = 800 # Mpa
#Pc = np.pi * Dm * f * ap * Kc /(60*np.power(10,6)*zeta_m)
#
# T: total torque from 3-phase ac motor
#T =(3.0/2.0)* k * Im * Bm # Im: max phase current; Bm: max magnetic 
