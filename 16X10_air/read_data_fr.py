#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:24:42 2021

@author: saquib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

path='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/test/tickle.txt'
pad_flag=True

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
    

##############################################################################

file=open(path, 'r')
window=5
lines=file.readlines()

data_lis=[] #this is a list that holds the entire dataset frame# X 160

for l in range(len(lines)): #loop through all the frames
    s=lines[l].replace(',,',',').split(',') #delete extra ","
    for i in range(len(s)-1): #loop through the 160 values in each frame
        if (s[i]==""):
            del s[i]
        elif (s[i]=="\n"):
            del s[i]
    nn=(list(map(int,s))) #creates a list with the cap values as separate variables
    if (len(nn)==160):
        data_lis.append(nn) #if there are 160 cap values, then add this frame to the master dataset

data_raw=np.array(data_lis)
baseline=np.mean(data_raw[0:3,:],axis=0)
data=data_raw-baseline
    

ext=[]
g_data=np.transpose(data[0:window,:])
#g_data=data[0:window,:].reshape(1,data.shape[1],window)
crop=data[0,:].reshape(1,-1)
signal_cnt=0;
base_cnt=0;
for frm in range(data.shape[0]): #loop through frames aka time in data
    cnt=0
    base_flag=False;
    signal_flag=True;
    for txl in range(data.shape[1]): #loop through taxels in data
        if (np.abs((data[frm,txl]))<3): #this is the threshold, under this it means 
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
        if ((pad_flag == True) and (signal_cnt==1)):
            #crop=np.append(crop,data[frm-3,:].reshape(1,-1),axis=0)
            crop=np.append(crop,data[frm-2,:].reshape(1,-1),axis=0)
            crop=np.append(crop,data[frm-1,:].reshape(1,-1),axis=0)
            crop=np.append(crop,data[frm,:].reshape(1,-1),axis=0)
            signal_cnt=3      
        else:
            crop=np.append(crop,data[frm,:].reshape(1,-1),axis=0)
        
        crop_t=np.transpose(crop)
    if ((signal_flag==True) and (signal_cnt>window)):

        g_data=np.dstack((g_data,crop_t[:,-window:]))

g2_data=np.zeros(shape=(g_data.shape[2],16,10,window))

for n_win in range(g_data.shape[2]):
    for n_frm in range(window):
        for y_frm in range(16):
            for x_frm in range(10):
                g2_data[n_win,y_frm,x_frm,n_frm]=g_data[(x_frm+10*y_frm),n_frm,n_win]

      #  g_data=np.concatenate((g_data,crop_t[:,-window:].reshape(1,crop_t.shape[0],window)),axis=0)


 

plt.figure(1)
for j in range(data.shape[1]):
    plt.plot(data[:,j])

#plt.figure(2)
#for j in range(crop.shape[1]):
#    plt.plot(crop[:,j])
#
#plt.figure(3)
#for j in range(g_data.shape[0]):
#    plt.plot(g_data[j,:,11])

D2_data=np.zeros(shape=(data.shape[0],16,10))
for n_ in range(data.shape[0]):
    for y_frm in range(16):
        for x_frm in range(10):
            D2_data[n_,y_frm,x_frm]=data[n_,(x_frm+10*y_frm)]


def animate(i):
    #data_a = D2_data[i,:,:] #select data range
#    plt.imshow(data_a, vmin=-0.02,vmax=0.56, cmap='gray')
    plt.imshow(D2_data[i,:,:], vmin=-1, vmax=1, cmap='bwr')

fig=plt.figure(1)
ani = animation.FuncAnimation(fig, animate, frames=data.shape[0], interval=30, repeat=True)

