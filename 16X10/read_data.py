#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:57:45 2020

@author: saquib
"""



import numpy as np
import matplotlib.pyplot as plt
path='/Users/saquib/Desktop/screenlog.txt'
file=open(path, 'r')
lines=file.readlines()

data_lis=[]

for l in range(len(lines)):
    s=lines[l].replace(',,',',').split(',')
    for i in range(len(s)-1):
        if (s[i]==""):
            del s[i]
        elif (s[i]=="\n"):
            del s[i]
    nn=(list(map(int,s)))
    if (len(nn)==160):
        data_lis.append(nn)

data=np.array(data_lis)
baseline=np.mean(data[0:3,:],axis=0)


ext=[]
crop=data[0,:].reshape(1,-1)
for j in range(data.shape[0]): #loop through t in data
    cnt=0
    for k in range(data.shape[1]): #loop through taxels in data
        if (np.abs((data[j,k]-baseline[k]))<3): #this is the threshold, under this it means 
            cnt=cnt+1                           #that the signal is baseline
            #print(cnt);
    ext.append(cnt);
    if (cnt<160): #if all the taxels are baseline then this will skip
        crop=np.append(crop,data[j,:].reshape(1,-1),axis=0)
        

#
plt.figure(1)
for j in range(data.shape[1]):
    plt.plot(data[:,j])

plt.figure(2)
for j in range(crop.shape[1]):
    plt.plot(crop[:,j])