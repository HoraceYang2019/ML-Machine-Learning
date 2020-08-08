import time
import DataPloter as dpl
 
class Timer(object):
    
    def __init__(self, period):
        self.period = period
        self.enFlag = True
    
    def run(self, fileName, xMax, yMax, sRate):
        s = dpl.Ploter(fileName, xMax, yMax, sRate)
        while (self.enFlag==True):
            s.showSignal()        
            time.sleep(self.period)  
    