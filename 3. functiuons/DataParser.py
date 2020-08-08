import csv

#----------------------------------------------------------------------------
class Parser(object):
    def parseRawFile(self, sFile, tFile, tNo): 
        fr = open(sFile, 'r', newline='\n')
        fw = open(tFile, 'w', newline='')
        tData= csv.writer(fw, delimiter=',')
    
        for s in csv.reader(fr):
            for i in tNo:
                temp = s[i].split(',')
                tData.writerow(temp) 
            
        print('Finished!')
        fr.close()
        fw.close()

# In[3]:
if __name__ == '__main__':
    sFile = './Source\\ForgingSignal.csv'
    tFile = './Source\\parsedSignal.csv'
    tNo = [0, 1]
    po = Parser()
    po.parseRawFile(sFile, tFile, tNo)

# In[]

fr = open(sFile, 'r', newline='\n')
fw = open(tFile, 'w', newline='')
tData = csv.writer(fw, delimiter=',')

   
for s in csv.reader(fr):
    for i in tNo:
        temp = s[i].split(',')
        tData.writerow(temp) 
            
print('Finished!')
fr.close()
fw.close()        