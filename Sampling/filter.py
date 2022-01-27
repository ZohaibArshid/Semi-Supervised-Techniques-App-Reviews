# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 23:15:02 2021

@author: Zobi Tanoli
"""

import pandas as pd
import numpy as np

xyz = pd.read_csv("filter.csv", index_col='Text', encoding="Latin1")
abc = pd.read_csv("sample.csv", index_col='Text', encoding="Latin1")

print(len(xyz))
print(len(abc))

for i in abc.index:
    if i in xyz.index:
        xyz.drop(i, axis=0, inplace=True)

print(len(xyz))

xyz.to_csv("filter.csv")