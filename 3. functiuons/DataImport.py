import DataParser as dps
import DataPloter as dpl    

sFile = 'RawData.csv'
tFile = 'Preprocess.csv'
tNo = [1]
xMax = 386
yMax = 30000
sRate = 1000  # sampling rate
period = 0.2
#------------------------------------------------------------------------------------------------        
if __name__ == '__main__':
    
    s =  dps.Parser() # new objects
    s.parseRawFile(sFile, tFile, tNo)
    
    l = dpl.Ploter()
    l.showSignal(tFile, xMax, yMax, sRate)