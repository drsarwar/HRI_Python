#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:29:58 2020

@author: saquib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('/Users/saquib/Downloads/CoST.csv')

loc=0
vn=np.zeros((8,8))
for i in range(8):
    for j in range(8):
        vn[i,j]=vv[loc]
        loc=loc+1
    