#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:57:45 2020

@author: saquib
"""



import numpy as np
import matplotlib.pyplot as plt
path='/Users/saquib/Desktop/screenlog.txt'


###########################################
#                                         #
# this part is only to extract baseline####
#                                         #
###########################################
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
b_data=data[0:5,:].reshape(1,data.shape[1],5)
for i in range(data.shape[0]-5):
    b_data=np.append(b_data,data[i:i+5,:].reshape(1,data.shape[1],-1),axis=0)
    




#####################33

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
g_data=data[0:5,:].reshape(1,data.shape[1],5)
crop=data[0,:].reshape(1,-1)
signal_cnt=0;
base_cnt=0;
for j in range(data.shape[0]): #loop through t in data
    cnt=0
    base_flag=False;
    signal_flag=True;
    for k in range(data.shape[1]): #loop through taxels in data
        if (np.abs((data[j,k]-baseline[k]))<3): #this is the threshold, under this it means 
            cnt=cnt+1                           #that the signal is baseline
            if (cnt==160):
                base_flag=True;
                signal_flag=False;
                base_cnt=base_cnt+1;
                if (base_cnt>2):
                    signal_cnt=0;

            #print(cnt);    
    ext.append(cnt);
    if (cnt<160): #if all the taxels are baseline then this will skip
        signal_flag=True;
        signal_cnt=signal_cnt+1
        base_cnt=0
        crop=np.append(crop,data[j,:].reshape(1,-1),axis=0)

    if ((signal_flag)&(signal_cnt>4)):
        g_data=np.append(g_data,crop[-5:,:].reshape(1,data.shape[1],-1),axis=0)
        
#
plt.figure(1)
for j in range(data.shape[1]):
    plt.plot(data[:,j])

plt.figure(2)
for j in range(crop.shape[1]):
    plt.plot(crop[:,j])

plt.figure(3)
for j in range(g_data.shape[1]):
    plt.plot(g_data[1,j,:])

