# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:18:09 2019

@author: ASUS
"""

1. read file and FFT from signal 
   FFT.py -> FF2.py 
   
2. transform function to object
   FFT.py -> DataPloter.py

3. parse file from csv file
   DataParser.py
   
4. use multiple files in a main file 
   DataImport.py calls DataParser and DataParser.py
   
5. read Parameter file
   Initial.py
   
6. merge multiple files
   DataMerger.py
   
7. integrate all files in a project
   Forging.py calls Initial, DataMerger, Parser, and Featurizer