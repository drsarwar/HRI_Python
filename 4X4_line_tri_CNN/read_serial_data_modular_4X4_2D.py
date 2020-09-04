#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:03:22 2020

@author: saquib
"""


import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

#adding some comment
class read_serial(object):
    def __init__(self):
        self.complete_data=[[],[]]
        self.cap_data=[]

    def read_data(self,path):
        f=list(open(path,'r'))
        
        num_taxel=16
        #self.complete_data=[[],[]]
        #self.cap_data=[]
        
        for i in range(len(f)):
            taxel=int((f[i][1:5]),base=2)
            self.complete_data[0].append(taxel)
            cap_value= float(f[i][7:13])
            self.complete_data[1].append(cap_value)
            if (len(self.cap_data)<num_taxel):
                self.cap_data.append([])
            else:
                self.cap_data[taxel].append(cap_value)

pa='/Users/saquib/Desktop/screenlog1.0'
X=read_serial()
X.read_data(pa)
data=np.array(X.cap_data)

for i in range(16):
    plt.plot(data[i,:])

data_t=np.transpose(data)
data_delta=np.zeros(shape=data_t.shape)
for j in range(data_t.shape[1]):
    data_delta[:,j]=data_t[:,j]-np.mean(data_t[:10,j])
#plt.figure(2)
#plt.plot(data[15,:])        
for i in range(16):
    plt.plot(data_delta[:,i],'o')    


#to make dataset we only want to keep the parts of the 
#data that correspond to a stimulus
#succesful crop for any stimulus over 1% change

data_ac=np.array([])
data_crop=np.zeros(shape=(1,16))
for k in range(data_delta.shape[0]):
    for l in range(data_delta.shape[1]):
        if data_delta[k,l]>0.01 and np.count_nonzero(data_ac == k)==0:
            data_crop=np.vstack((data_crop,data_delta[k,:].reshape(1,-1)))
            data_ac=np.append(data_ac,k)
    
for i in range(16):
    plt.plot(data_crop[:,i])    
    


data2D=np.zeros(shape=(4,4,data_t.shape[0]))
#mapping for 4X4 board 3mm with arduino mega
for i in range(data_t.shape[0]):
    data2D[0,0,i]=data_t[i,5]
    data2D[0,1,i]=data_t[i,4]
    data2D[0,2,i]=data_t[i,7]
    data2D[0,3,i]=data_t[i,6]
    data2D[1,0,i]=data_t[i,1]
    data2D[1,1,i]=data_t[i,0]
    data2D[1,2,i]=data_t[i,3]
    data2D[1,3,i]=data_t[i,2]
    data2D[2,0,i]=data_t[i,13]
    data2D[2,1,i]=data_t[i,12]
    data2D[2,2,i]=data_t[i,15]
    data2D[2,3,i]=data_t[i,14]
    data2D[3,0,i]=data_t[i,9]
    data2D[3,1,i]=data_t[i,8]
    data2D[3,2,i]=data_t[i,11]
    data2D[3,3,i]=data_t[i,10]
    
#can also do it this way
#data2D[0,0,:]=data_t[:,5]
#data2D[0,1,:]=data_t[:,4]
#data2D[0,2,:]=data_t[:,7]
#data2D[0,3,:]=data_t[:,6]
#data2D[1,0,:]=data_t[:,1]
#data2D[1,1,:]=data_t[:,0]
#data2D[1,2,:]=data_t[:,3]
#data2D[1,3,:]=data_t[:,2]
#data2D[2,0,:]=data_t[:,13]
#data2D[2,1,:]=data_t[:,12]
#data2D[2,2,:]=data_t[:,15]
#data2D[2,3,:]=data_t[:,14]
#data2D[3,0,:]=data_t[:,9]
#data2D[3,1,:]=data_t[:,8]
#data2D[3,2,:]=data_t[:,11]
#data2D[3,3,:]=data_t[:,10]



data2D_delta=np.zeros(shape=(4,4,data_t.shape[0]))
#mapping for 4X4 board 3mm with arduino mega
for i in range(data_delta.shape[0]):
    data2D_delta[0,0,i]=data_delta[i,5]
    data2D_delta[0,1,i]=data_delta[i,4]
    data2D_delta[0,2,i]=data_delta[i,7]
    data2D_delta[0,3,i]=data_delta[i,6]
    data2D_delta[1,0,i]=data_delta[i,1]
    data2D_delta[1,1,i]=data_delta[i,0]
    data2D_delta[1,2,i]=data_delta[i,3]
    data2D_delta[1,3,i]=data_delta[i,2]
    data2D_delta[2,0,i]=data_delta[i,13]
    data2D_delta[2,1,i]=data_delta[i,12]
    data2D_delta[2,2,i]=data_delta[i,15]
    data2D_delta[2,3,i]=data_delta[i,14]
    data2D_delta[3,0,i]=data_delta[i,9]
    data2D_delta[3,1,i]=data_delta[i,8]
    data2D_delta[3,2,i]=data_delta[i,11]
    data2D_delta[3,3,i]=data_delta[i,10]



data2D_crop=np.zeros(shape=(4,4,data_t.shape[0]))
#mapping for 4X4 board 3mm with arduino mega
for i in range(data_crop.shape[0]):
    data2D_crop[0,0,i]=data_crop[i,5]
    data2D_crop[0,1,i]=data_crop[i,4]
    data2D_crop[0,2,i]=data_crop[i,7]
    data2D_crop[0,3,i]=data_crop[i,6]
    data2D_crop[1,0,i]=data_crop[i,1]
    data2D_crop[1,1,i]=data_crop[i,0]
    data2D_crop[1,2,i]=data_crop[i,3]
    data2D_crop[1,3,i]=data_crop[i,2]
    data2D_crop[2,0,i]=data_crop[i,13]
    data2D_crop[2,1,i]=data_crop[i,12]
    data2D_crop[2,2,i]=data_crop[i,15]
    data2D_crop[2,3,i]=data_crop[i,14]
    data2D_crop[3,0,i]=data_crop[i,9]
    data2D_crop[3,1,i]=data_crop[i,8]
    data2D_crop[3,2,i]=data_crop[i,11]
    data2D_crop[3,3,i]=data_crop[i,10]



def animate(i):
    data_a = data2D_delta[:,:,i] #select data range
#    plt.imshow(data_a, vmin=-0.02,vmax=0.56, cmap='gray')
    plt.imshow(data_a, vmin=-0.3, vmax=0.8, cmap='gray')

fig=plt.figure(1)
ani = animation.FuncAnimation(fig, animate, frames=data_t.shape[0], interval=300, repeat=True)


    

