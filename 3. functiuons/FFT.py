#===============================================================================
# for Rasbian: 
# sudo apt-get install python3-pip
# sudo pip3 install numpy
# sudo apt-get install python3-matplotlib
#----------------------------------------------
# sudo apt-get upgrade
#===============================================================================
#===============================================================================
# for Windows:
# cd ./python/python36/scripts/
# pip install numpy
#===============================================================================
 
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

N = 2000      # no. of data points
sRate = 1000  # sampling rate
T = 1/sRate   # sampling period

t = np.linspace(0.0, N*T, N) #

w0 = 0.1    # noise weighting
bias = 0
f1 = 250.0 # in Hz
w1 = 1    # weight     t
f2 = 100.0
w2 = 2
x = bias + w1*np.sin(f1*2.0*np.pi*t) + w2*np.sin(f2*2.0*np.pi*t) + w0*np.random.random(N)
y = fft(x)
f = np.linspace(0.0, 1.0/(2.0*T), N/2)

plt.figure(1)
plt.subplot(211)  # plot subplot
plt.plot(t, x, lw=2)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid(True)
plt.subplot(212)
plt.plot(f, 2.0/N*abs(y.real[0:int(N/2)])) 
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.grid(True)
plt.show()