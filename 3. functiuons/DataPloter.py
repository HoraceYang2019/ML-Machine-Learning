import csv
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy.fft import fft  

class Ploter(object):
    def showSignal(self, fileName, xMax, yMax, sRate):
        fp = open(fileName, 'r', newline='')
       
        N = xMax
        T = 1/sRate   # sampling period
        plt.ion()               # interactive mode open
        fig = plt.figure()      # assign figure             
        tPlot = fig.add_subplot(211)        
        t = np.linspace(0.0, N*T, N) 
        ty = np.linspace(0, yMax, N)  # default     
        tLine, = tPlot.plot(t, ty, 'r-')
        plt.title('Forging Force')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (ms)') 
        plt.grid(True)        
          
        fPlot = fig.add_subplot(212)
        f = np.linspace(0.0, 1.0/(2.0*T), N/2)
        fy = np.linspace(0.0, 1000, N/2)
        fLine, = fPlot.plot(f, fy, 'b-')
        plt.ylabel('Amplitude')
        plt.xlabel('Frequency (Hz)')
        plt.grid(True)
          
        plt.show()
      
        for sData in csv.reader(fp): 
            tLine.set_ydata(sData) # set y data of time
                          
            fy = fft(sData)        # get y data
            fLine.set_ydata(2.0/N*abs(fy.real[0:int(N/2)]))
                     
            fig.canvas.draw()    # draw canvas of the figure  
            plt.pause(0.001)                         
            time.sleep(0.2)
  
            #os.system("pause")
        fp.close()         