#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:39:32 2020

@author: saquib
"""



import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
from tensorflow import keras
from scipy import signal
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
from keras.datasets import reuters
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from keras.layers import Conv3D
from keras.layers import Flatten
from keras.utils import plot_model
import sys, serial
from keras.layers import BatchNormalization


def extract_data(d_file):
    file=open(d_file, 'r')
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
    
    data_raw=np.array(data_lis)
    baseline=np.mean(data_raw[0:3,:],axis=0)
    data=data_raw-baseline
    
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
            if (np.abs((data_raw[j,k]-baseline[k]))<3): #this is the threshold, under this it means 
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
    
    g_data_scaled=g_data/512
    g_data_2D=g_data_scaled.reshape(g_data.shape[0],16,10,g_data.shape[2])    
    
    return(crop,g_data,g_data_2D)

def extract_baseline(d_file):
    file=open(d_file, 'r')
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
    
    data_raw=np.array(data_lis)
    baseline=np.mean(data_raw[0:3,:],axis=0)
    data=data_raw-baseline
    
    b_data=data[0:5,:].reshape(1,data.shape[1],5)
    for i in range(data.shape[0]-5):
        b_data=np.append(b_data,data[i:i+5,:].reshape(1,data.shape[1],-1),axis=0)
    
    b_data_scaled=b_data/512
    b_data_2D=b_data_scaled.reshape(b_data.shape[0],16,10,b_data.shape[2])    
   
    return (b_data,b_data_2D)

baseline_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10/baseline.txt'
stroke_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10/stroke.txt'
massage_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10/massage.txt'

#paths for test data
baseline_test_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10/baseline_test.txt'
stroke_test_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10/stroke_test.txt'
massage_test_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10/massage_test.txt'


g_base,g2_base=extract_baseline(baseline_file) #import x data for baseline
c_stroke,g_stroke,g2_stroke=extract_data(stroke_file) #import x data for stroke
c_massage,g_massage,g2_massage=extract_data(massage_file) #import x data for massage
x_total=np.concatenate((g2_base[0:1500,:,:,:], g2_stroke),axis=0)
x_total=np.concatenate((x_total,g2_massage), axis=0)

#acquiring test data
g_base_t,g2_base_t=extract_baseline(baseline_test_file) #import x test data for baseline
c_stroke_t,g_stroke_t,g2_stroke_t=extract_data(stroke_test_file) #import x data for stroke
c_massage_t,g_massage_t,g2_massage_t=extract_data(massage_test_file) #import x data for massage
x_total_t=np.concatenate((g2_base_t[0:200,:,:,:], g2_stroke_t[0:200,:,:,:]),axis=0)
x_total_t=np.concatenate((x_total_t,g2_massage_t), axis=0)




y_base=np.zeros(g2_base[0:1500,:,:,:].shape[0]) #creating y labels for baseline
y_stroke=np.ones(g2_stroke.shape[0]) #creating y labels for stroke
y_massage=2*np.ones(g2_massage.shape[0]) #creating y labels for massage
y_total=np.append(y_base,y_stroke) #creating final y by adding base and stroke
y_total=np.append(y_total,y_massage) #adding in massage

y_all = to_categorical(y_total) #one hot encoding


#generate test labels

y_base_t=np.zeros(g2_base_t[0:200,:,:,:].shape[0]) #creating y labels for baseline
y_stroke_t=np.ones(g2_stroke_t[0:200,:,:,:].shape[0]) #creating y labels for stroke
y_massage_t=2*np.ones(g2_massage_t.shape[0]) #creating y labels for massage
y_total_t=np.append(y_base_t,y_stroke_t) #creating final y by adding base and stroke
y_total_t=np.append(y_total_t,y_massage_t) #adding in massage

y_all_t = to_categorical(y_total_t) #one hot encoding



X_train, X_test, y_train, y_test = train_test_split(
         x_total, y_all, test_size=0.2, random_state=1, stratify=y_total)


model=Sequential()

model.add(Conv3D(30,
                 kernel_size=(3,3,3),
                 strides=(1, 1, 1),
                 activation='relu',
                 input_shape=(16, 10, 5, 1),
                 padding='same'))

model.add(Conv3D(30,
                 kernel_size=(3,3,3),
                 strides=(1, 1, 1),
                 activation='relu',
                 padding='same'))

#model.add(Conv2D(10,
#                 kernel_size=2,
#                 activation='relu',
#                 padding='same'))
#model.add(Conv2D(10,
#                 kernel_size=2,
#                 activation='relu',
#                 padding='same'))
model.add(Flatten())
#model.add(Dense(784, 
#                activation='relu',
#                ))

#model.add(Dense(64, 
#                activation='relu'))
model.add(Dense(320, 
                activation='relu'))
#model.add(BatchNormalization())

model.add(Dense(160, 
                activation='relu'))
#model.add(BatchNormalization())


model.add(Dense(80, 
                activation='relu'))

model.add(Dense(3, 
                activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#using validation data
#history=model.fit(X_train.reshape(X_train.shape[0],16,10,5,1), y_train,
#                  epochs=12,
#                  validation_data = (X_test.reshape(X_test.shape[0],16,10,5,1), y_test))


#using test data

history=model.fit(X_train.reshape(X_train.shape[0],16,10,5,1), y_train,
                  epochs=12,
                  validation_data = (x_total_t.reshape(x_total_t.shape[0],16,10,5,1), y_all_t))


#
#

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.ylim(0,1.1)
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.ylim(0,1.1)
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





strPort = '/dev/cu.usbserial-FTBS2SJQ'
ser = serial.Serial(strPort, baudrate=115200)

#
#fig,ax = plt.subplots(1,1)
#image = data2D_delta_test[0,:,:]
#im = ax.imshow(image,vmin=-0.05, vmax=0.0002,cmap='gray')

#this part reads 5 sets of data and then averages them 
#in order to get the baseline

rt_base=[]

for j in range(5):
    line = str(ser.readline())
    #print(line)
    s=line.replace(',,',',').split(',')
    for i in range(len(s)-1):
        if (s[i]==""):
            del s[i]
        elif (s[i]=="b'"):
            del s[i]
        elif (s[i]=="\n'"):
            del s[i]
        elif (s[i]=="\\n'"):
            del s[i]
    nn=(list(map(int,s)))
    if (len(nn)==160):
        rt_base.append(nn)

rt_baseline=np.mean(np.array(rt_base),axis=0)


while True:
    
    rt_data_l=[]
    for k in range(5):
        line = str(ser.readline())
        #print(line)
        s=line.replace(',,',',').split(',')
        for i in range(len(s)-1):
            if (s[i]==""):
                del s[i]
            elif (s[i]=="b'"):
                del s[i]
            elif (s[i]=="\n'"):
                del s[i]
            elif (s[i]=="\\n'"):
                del s[i]
        nn=(list(map(int,s)))
        if (len(nn)==160):
            rt_data_l.append(nn)
    
    rt_data=(np.array(rt_data_l)-rt_baseline).reshape(1,16,10,5,1)/512
    
    result=model.predict(rt_data)
    if (np.argmax(result)==0):
        print("baseline")
    elif(np.argmax(result)==1):
        print("stroke")
    elif(np.argmax(result)==2):
        print("massage")
    
#    
#    
#    data_raw=np.array(data_lis)
#    baseline=np.mean(data_raw[0:3,:],axis=0)
#    data=data_raw-baseline
#    
#    b_data=data[0:5,:].reshape(1,data.shape[1],5)
#    for i in range(data.shape[0]-5):
#        b_data=np.append(b_data,data[i:i+5,:].reshape(1,data.shape[1],-1),axis=0)
#    
#    b_data_scaled=b_data/512
#    b_data_2D=b_data_scaled.reshape(b_data.shape[0],16,10,b_data.shape[2])    
# 
#    
#    
    
#
#while True:
#    # start = time.process_time()
#    line = ser.readline()
#    data=str(line)
#    if (data[2]=='(') and (data[7]==')'):
#        indx=int(data[3:7],2)
#        cnt=cnt+1
#       
#        if indx==0:
#            stat[0]=float(data[8:13])-base_cap[0]
#        elif indx==1:
#            stat[1]=float(data[8:13])-base_cap[1]
#        elif indx==2:
#            stat[2]=float(data[8:13])-base_cap[2]
#        elif indx==3:
#            stat[3]=float(data[8:13])-base_cap[3]
#        elif indx==4:
#            stat[4]=float(data[8:13])-base_cap[4]
#        elif indx==5:
#            stat[5]=float(data[8:13])-base_cap[5]
#        elif indx==6:
#            stat[6]=float(data[8:13])-base_cap[6]
#        elif indx==7:
#            stat[7]=float(data[8:13])-base_cap[7]
#        elif indx==8:
#            stat[8]=float(data[8:13])-base_cap[8]
#        elif indx==9:
#            stat[9]=float(data[8:13])-base_cap[9]
#        elif indx==10:
#            stat[10]=float(data[8:13])-base_cap[10]
#        elif indx==11:
#            stat[11]=float(data[8:13])-base_cap[11]
#        elif indx==12:
#            stat[12]=float(data[8:13])-base_cap[12]
#        elif indx==13:
#            stat[13]=float(data[8:13])-base_cap[13]
#        elif indx==14:
#            stat[14]=float(data[8:13])-base_cap[14]
#        elif indx==15:
#            stat[15]=float(data[8:13])-base_cap[15]
#            
#        if cnt==15:
#            #print(stat)
#            
#            #------------------------#
#            #using baseline correction
##            
##                        
##            stat_delta=np.zeros(shape=stat.shape)
##            for j in range(stat.shape[0]):
##                stat_delta[j]=stat[j]-np.mean(data_t[:10,j])
##    
##            stat=stat_delta
#            #----------------------#
#            
#            signal2D=np.zeros(shape=(4,4))
#            #mapping for 4X4 board 3mm with arduino mega
#            
#            signal2D[0,0]=stat[5]
#            signal2D[0,1]=stat[4]
#            signal2D[0,2]=stat[7]
#            signal2D[0,3]=stat[6]
#            signal2D[1,0]=stat[1]
#            signal2D[1,1]=stat[0]
#            signal2D[1,2]=stat[3]
#            signal2D[1,3]=stat[2]
#            signal2D[2,0]=stat[13]
#            signal2D[2,1]=stat[12]
#            signal2D[2,2]=stat[15]
#            signal2D[2,3]=stat[14]
#            signal2D[3,0]=stat[9]
#            signal2D[3,1]=stat[8]
#            signal2D[3,2]=stat[11]
#            signal2D[3,3]=stat[10]
#
#            if np.argmax(model.predict(signal2D.reshape(1,4,4,1)),axis=1)==0:
#                print('baseline')
#            elif np.argmax(model.predict(signal2D.reshape(1,4,4,1)),axis=1)==1:
#                print('proximity')
#            elif np.argmax(model.predict(signal2D.reshape(1,4,4,1)),axis=1)==2:
#                print('touch')
#            # elif clf.predict(stat.reshape(1,16))==3:
#            #     print('proximity')
#            else:
#                print('unrecognized')
#            #print(clf.predict(stat.reshape(1,4)))
#            # print((time.process_time() - start)*1000)
#            image = signal2D
#            im.set_data(image)
#            fig.canvas.draw_idle()
#            plt.pause(0.001)
#            cnt=0
#    else:
#            continue
#  
#
