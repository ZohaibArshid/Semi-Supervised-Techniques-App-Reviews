# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 22:52:10 2021

@author: Zobi Tanoli
"""

import pandas as pd

csvfile = pd.read_csv('filter.csv', encoding= 'Latin1')
smple= csvfile.sample(300)
print(len(csvfile))
print(len(smple))

#print((smple))




smple.to_csv("sample30.csv")