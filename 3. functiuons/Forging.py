# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:41:59 2017

@author: Horace Yang
"""
import Initial as ini
import DataMerger as dmr
import Parser as par
import Featurizer as far

# In[11]:
#------------------------------------------------------------------------------------------------        
if __name__ == '__main__':
       
    parm = ini.parm  # access parameters from intiial file
    
   # Step 1: merge multiple data files into a data file
    mr = dmr.DataMerger(parm['sourDir'])
    s1 = mr.fit_transform(parm['sFile']) 
    
   # Step 2: parse data 
    po = par.Parser(index = parm['tNo'], timeTag = parm['timeTag'], logShow = parm['logShow'])
    s2 = po.fit_transform(s1) # sheet name of the target file, return raw data list with time tag
    # smp = np.array(s1).astype(int)
    
   # Step 3: extract features 
    fo = far.Featurizer(timeTag = parm['timeTag'], fFile = parm['fFile'], ntype = parm['nType'], 
                     grp = parm['nGroup'], band = parm['fBand'], iOB = parm['fIOB'], 
                     mode = parm['dMode'], dur = parm['Durations'], infFactor = parm['inflection factor'],
                     sheet = '', logShow = parm['logShow']) 
    
    # fur.getInflectionPt(smp[0])
    s3 = fo.fit_transform(s2) # new miner object

