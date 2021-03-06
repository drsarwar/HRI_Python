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
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.utils import plot_model
import sys, serial
from keras.layers import BatchNormalization
from keras.layers import ConvLSTM2D
from keras.callbacks import EarlyStopping


window=10
pad_flag=False

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
    
    #window=5
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
                #crop=np.append(crop,data[frm-4,:].reshape(1,-1),axis=0)
                #crop=np.append(crop,data[frm-3,:].reshape(1,-1),axis=0)
                crop=np.append(crop,data[frm-2,:].reshape(1,-1),axis=0)
                crop=np.append(crop,data[frm-1,:].reshape(1,-1),axis=0)
                crop=np.append(crop,data[frm,:].reshape(1,-1),axis=0)
                signal_cnt=3  
                #signal_cnt=5
            else:
                crop=np.append(crop,data[frm,:].reshape(1,-1),axis=0)
            
            crop_t=np.transpose(crop)
        if ((signal_flag==True) and (signal_cnt>window)):
            g_data=np.dstack((g_data,crop_t[:,-window:]))
    
    g2_data=np.zeros(shape=(g_data.shape[2]-1,window,16,10))
    
    for n_win in range(g_data.shape[2]-1):
        for n_frm in range(window):
            for y_frm in range(16):
                for x_frm in range(10):
                    g2_data[n_win,n_frm,y_frm,x_frm]=g_data[(x_frm+10*y_frm),n_frm,n_win+1]
    
    g_data_2D=g2_data/256
    #g_data_2D=g_data_scaled.reshape(g_data.shape[0],16,10,g_data.shape[2])    
    
    return(crop,g_data,g_data_2D)

def extract_baseline(d_file):
    file_b=open(d_file, 'r')
    lines_b=file_b.readlines()
    
    data_lis_b=[]
    
    for l in range(len(lines_b)):
        s_b=lines_b[l].replace(',,',',').split(',')
        for i in range(len(s_b)-1):
            if (s_b[i]==""):
                del s_b[i]
            elif (s_b[i]=="\n"):
                del s_b[i]
        nn_b=(list(map(int,s_b)))
        if (len(nn_b)==160):
            data_lis_b.append(nn_b)
    
    data_raw_b=np.array(data_lis_b)
    baseline_b=np.mean(data_raw_b[0:3,:],axis=0)
    data_b=data_raw_b-baseline_b
    
    b_data=data_b[0:window,:]
    for i in range(data_b.shape[0]-window):
        b_data=np.dstack((b_data,data_b[i:i+window,:]))
    
    b2_data=np.zeros(shape=(b_data.shape[2],window,16,10))
    
    for n_win in range(b_data.shape[2]):
        for n_frm in range(window):
            for y_frm in range(16):
                for x_frm in range(10):
                    b2_data[n_win,n_frm,y_frm,x_frm]=b_data[n_frm,(x_frm+10*y_frm),n_win]
    
    b_data_2D=b2_data/256
    
   
    return (b_data,b_data_2D)


dir_path='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air'
baseline_file=dir_path+'/baseline.txt'
air_stroke_file=dir_path+'/air_stroke.txt'
light_stroke_file=dir_path+'/light_stroke.txt'
hard_stroke_file=dir_path+'/hard_stroke.txt'
tickle_file=dir_path+'/tickle.txt'
hit_file=dir_path+'/hit.txt'
hover_file=dir_path+'/hover.txt'
light_touch_file=dir_path+'/light_touch.txt'

#paths for test data
baseline_file_t=dir_path+'/test/baseline.txt'
air_stroke_file_t=dir_path+'/test/air_stroke.txt'
light_stroke_file_t=dir_path+'/test/light_stroke.txt'
hard_stroke_file_t=dir_path+'/test/hard_stroke.txt'
tickle_file_t=dir_path+'/test/tickle.txt'
hit_file_t=dir_path+'/test/hit.txt'
hover_file_t=dir_path+'/test/hover.txt'
light_touch_file_t=dir_path+'/test/light_touch.txt'

#acquiring data
g_base,g2_base=extract_baseline(baseline_file) #import x data for baseline
c_air_stroke,g_air_stroke,g2_air_stroke=extract_data(air_stroke_file) #import x data for air stroke
c_light_stroke,g_light_stroke,g2_light_stroke=extract_data(light_stroke_file) #import x data for light stroke
c_hard_stroke,g_hard_stroke,g2_hard_stroke=extract_data(hard_stroke_file)
c_tickle,g_tickle,g2_tickle=extract_data(tickle_file)
#c_hit,g_hit,g2_hit=extract_data(hit_file)
c_hover,g_hover,g2_hover=extract_data(hover_file)
c_light_touch,g_light_touch,g2_light_touch=extract_data(light_touch_file)

x_total=np.concatenate((g2_base[:500,:,:,:], g2_air_stroke),axis=0)
x_total=np.concatenate((x_total,g2_light_stroke), axis=0)
x_total=np.concatenate((x_total,g2_hard_stroke), axis=0)
x_total=np.concatenate((x_total,g2_tickle[:,:,:,:]), axis=0)
#x_total=np.concatenate((x_total,g2_hit), axis=0)
x_total=np.concatenate((x_total,g2_hover[:1000,:,:,:]), axis=0)
x_total=np.concatenate((x_total,g2_light_touch[:,:,:,:]), axis=0)


#generate training labels

y_base=np.zeros(g2_base[:500,:,:,:].shape[0]) #creating y labels for baseline
y_air_stroke=np.ones(g2_air_stroke.shape[0]) #creating y labels for stroke
y_light_stroke=2*np.ones(g2_light_stroke.shape[0]) #creating y labels for massage
y_hard_stroke=3*np.ones(g2_hard_stroke.shape[0])
y_tickle=4*np.ones(g2_tickle[:,:,:,:].shape[0])
#y_hit=5*np.ones(g2_hit.shape[0])
y_hover=5*np.ones(g2_hover[:1000,:,:,:].shape[0])
y_light_touch=6*np.ones(g2_light_touch[:,:,:,:].shape[0])


y_total=np.append(y_base,y_air_stroke) #creating final y by adding base and stroke
y_total=np.append(y_total,y_light_stroke) #adding in massage
y_total=np.append(y_total,y_hard_stroke)
y_total=np.append(y_total,y_tickle)
#y_total=np.append(y_total,y_hit)
y_total=np.append(y_total,y_hover)
y_total=np.append(y_total,y_light_touch)


y_all = to_categorical(y_total) #one hot encoding


#acquiring test data
g_base_t,g2_base_t=extract_baseline(baseline_file_t) #import x data for baseline
c_air_stroke_t,g_air_stroke_t,g2_air_stroke_t=extract_data(air_stroke_file_t) #import x data for air stroke
c_light_stroke_t,g_light_stroke_t,g2_light_stroke_t=extract_data(light_stroke_file_t) #import x data for light stroke
c_hard_stroke_t,g_hard_stroke_t,g2_hard_stroke_t=extract_data(hard_stroke_file_t)
c_tickle_t,g_tickle_t,g2_tickle_t=extract_data(tickle_file_t)
#c_hit_t,g_hit_t,g2_hit_t=extract_data(hit_file_t)
c_hover_t,g_hover_t,g2_hover_t=extract_data(hover_file_t)
c_light_touch_t,g_light_touch_t,g2_light_touch_t=extract_data(light_touch_file_t)



x_total_t=np.concatenate((g2_base_t[:100,:,:,:], g2_air_stroke_t),axis=0)
x_total_t=np.concatenate((x_total_t,g2_light_stroke_t), axis=0)
x_total_t=np.concatenate((x_total_t,g2_hard_stroke_t[:,:,:,:]), axis=0)
x_total_t=np.concatenate((x_total_t,g2_tickle_t), axis=0)
#x_total_t=np.concatenate((x_total_t,g2_hit_t), axis=0)
x_total_t=np.concatenate((x_total_t,g2_hover_t), axis=0)
x_total_t=np.concatenate((x_total_t,g2_light_touch_t), axis=0)



#generate test labels
#
y_base_t=np.zeros(g2_base_t[:100,:,:,:].shape[0]) #creating y labels for baseline
y_air_stroke_t=np.ones(g2_air_stroke_t.shape[0]) #creating y labels for stroke
y_light_stroke_t=2*np.ones(g2_light_stroke_t.shape[0]) #creating y labels for massage
y_hard_stroke_t=3*np.ones(g2_hard_stroke_t[:,:,:,:].shape[0])
y_tickle_t=4*np.ones(g2_tickle_t.shape[0])
#y_hit_t=5*np.ones(g2_hit_t.shape[0])
y_hover_t=5*np.ones(g2_hover_t.shape[0])
y_light_touch_t=6*np.ones(g2_light_touch_t.shape[0])


y_total_t=np.append(y_base_t,y_air_stroke_t) #creating final y by adding base and stroke
y_total_t=np.append(y_total_t,y_light_stroke_t) #adding in massage
y_total_t=np.append(y_total_t,y_hard_stroke_t)
y_total_t=np.append(y_total_t,y_tickle_t)
#y_total_t=np.append(y_total_t,y_hit_t)
y_total_t=np.append(y_total_t,y_hover_t)
y_total_t=np.append(y_total_t,y_light_touch_t)


y_all_t = to_categorical(y_total_t) #one hot encoding


X_train, X_test, y_train, y_test = train_test_split(
         x_total, y_all, test_size=0.1, random_state=1, stratify=y_total)


e_stop=EarlyStopping(
        monitor='accuracy',
        patience=0,
        verbose=1,
        min_delta=0.001,
    #    restore_best_weights = True,
        mode='max')

model=Sequential()


model.add(TimeDistributed(
                        Conv2D(30,
                               kernel_size=(3,3),
                               activation='relu',
                               input_shape=(window,16,10,1),
                               padding='same')))

model.add(TimeDistributed(
                        Conv2D(30,
                               kernel_size=(3,3),
                               activation='relu',
                               padding='same')))

model.add(TimeDistributed(
                        Flatten()))
#model.add(LSTM(160,
#               activation='relu',
#               return_sequences=True))

model.add(LSTM(160,
               activation='relu',
               return_sequences=False))


model.add(Dense(320, 
                activation='relu'))
#model.add(BatchNormalization())

model.add(Dense(160, 
                activation='relu'))
#model.add(BatchNormalization())


model.add(Dense(80, 
                activation='relu'))



model.add(Dense(7, 
                activation='softmax'))
#opt=keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#
#history=model.fit(X_train.reshape(X_train.shape[0],5,16,10,1), y_train,
#                  epochs=12,
#                  validation_data = (X_test.reshape(X_test.shape[0],5,16,10,1), y_test))
#using test data

history=model.fit(X_train.reshape(X_train.shape[0],window,16,10,1), y_train,
                  epochs=30,
                  validation_data = (x_total_t.reshape(x_total_t.shape[0],window,16,10,1), y_all_t),
                  callbacks=[e_stop])

model.summary()

test_result = model.evaluate(x_total_t.reshape(x_total_t.shape[0],window,16,10,1), y_all_t)
htry=history.history

pred=model.predict(x_total_t.reshape(x_total_t.shape[0],window,16,10,1))

y_true=np.array(tf.argmax(y_all_t,axis=1))
y_pred=np.array(tf.argmax(pred, axis=1))
cm=np.array(tf.math.confusion_matrix(y_true, y_pred))




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



#--------------------------------------------------------------
#This part of the code is for real time recognition
#--------------------------------------------------------------


strPort = '/dev/cu.usbserial-FTBS2SJQ'
ser = serial.Serial(strPort, baudrate=115200)

#
#fig,ax = plt.subplots(1,1)
#image = data2D_delta_test[0,:,:]
#im = ax.imshow(image,vmin=-0.05, vmax=0.0002,cmap='gray')
#---------------------------------------------------------
#this part reads 5 sets of data and then averages them 
#in order to get the baseline
#---------------------------------------------------------
rt_base=[]

for j in range(5):
    line = str(ser.readline())
    #print(line)
    s=line.replace(',,',',').split(',')
    #this loop is to delete junk variables that come through the port
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

#----------------------------------------------------------
#this creates the first frame of window data points
#----------------------------------------------------------
rt_frame_l=[]
for k in range(window):
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
        data_after_base=nn-rt_baseline
        rt_frame_l.append(data_after_base)

#----------------------------------------------------------

while True:
    #read current frame
    
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
        del rt_frame_l[0]    
        data_after_base=nn-rt_baseline
        rt_frame_l.append(data_after_base)   
    
    
    #rt_data=(np.array(rt_frame_l)-rt_baseline).reshape(1,16,10,window,1)/256
    g_data_rt=np.array(rt_frame_l)
    g2_data_rt=np.zeros(shape=(1,window,16,10,1))
    

    for n_frm in range(window):
        for y_frm in range(16):
            for x_frm in range(10):
                g2_data_rt[0,n_frm,y_frm,x_frm]=g_data_rt[n_frm,(x_frm+10*y_frm)]
    
    rt_data=g2_data_rt/256    
    
    
    result=model.predict(rt_data)
    if (np.argmax(result)==0):
        print("baseline")
    elif(np.argmax(result)==1):
        print("air_stroke")
    elif(np.argmax(result)==2):
        print("light_stroke")
    elif(np.argmax(result)==3):
        print("hard_stroke")
    elif(np.argmax(result)==4):
        print("tickle")
    elif(np.argmax(result)==5):
        print("hover")
    elif(np.argmax(result)==6):
        print("light_touch")