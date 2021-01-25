#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:39:32 2020

@author: saquib
"""



import numpy as np
import matplotlib.pyplot as plt
# import time
# import matplotlib.animation as animation
# from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from scipy import signal
# from keras.datasets import mnist
# import tensorflow as tf
# from keras.datasets import reuters
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from tensorflow.keras import regularizers
# import matplotlib.pyplot as plt
# from keras.layers import Conv3D
# from keras.layers import Flatten
# from keras.utils import plot_model
# import sys, serial
# from keras.layers import BatchNormalization
# from sklearn.metrics import plot_confusion_matrix


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
    
    window=5
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
            if (np.abs((data[frm,txl]))<4): #this is the threshold, under this it means 
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
    
    g_data_2D=g2_data/256
    #g_data_2D=g_data_scaled.reshape(g_data.shape[0],16,10,g_data.shape[2])    
    
    return(crop,g_data,g_data_2D)

def extract_baseline(d_file):
    file=open(d_file, 'r')
    lines=file.readlines()
    window=5
    data_lis=[]
    
    for l in range(len(lines)): #loop through frames aka time 
        s=lines[l].replace(',,',',').split(',')
        for i in range(len(s)-1):
            if (s[i]==""):
                del s[i]
            elif (s[i]=="\n"):
                del s[i]
        nn=(list(map(int,s))) #map the data to a list of cap magnitudes
        if (len(nn)==160):
            data_lis.append(nn) #if all 160 is present, treat it as a valid frame and append
    
    data_raw=np.array(data_lis)
    baseline=np.mean(data_raw[0:3,:],axis=0)
    data=data_raw-baseline
    
    #b_data=data[0:window,:].reshape(1,data.shape[1],window)
    b_data=data[0:window,:]
    for i in range(data.shape[0]-window):
        b_data=np.dstack((b_data,data[i:i+window,:]))
    
    b2_data=np.zeros(shape=(b_data.shape[2],16,10,window))
    
    for n_win in range(b_data.shape[2]):
        for n_frm in range(window):
            for y_frm in range(16):
                for x_frm in range(10):
                    b2_data[n_win,y_frm,x_frm,n_frm]=b_data[n_frm,(x_frm+10*y_frm),n_win]
    
    b_data_2D=b2_data/256
   
    return (b_data,b_data_2D)

baseline_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/baseline.txt'
air_stroke_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/air_stroke.txt'
light_stroke_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/light_stroke.txt'
hard_stroke_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/hard_stroke.txt'
tickle_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/tickle.txt'
hit_file='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/hit.txt'

#paths for test data
baseline_file_t='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/test/baseline.txt'
air_stroke_file_t='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/test/air_stroke.txt'
light_stroke_file_t='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/test/light_stroke.txt'
hard_stroke_file_t='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/test/hard_stroke.txt'
tickle_file_t='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/test/tickle.txt'
hit_file_t='/Users/saquib/Documents/Research/HRI/HRI_Python/16X10_air/test/hit.txt'

#acquiring data
g_base,g2_base=extract_baseline(baseline_file) #import x data for baseline
c_air_stroke,g_air_stroke,g2_air_stroke=extract_data(air_stroke_file) #import x data for air stroke
c_light_stroke,g_light_stroke,g2_light_stroke=extract_data(light_stroke_file) #import x data for light stroke
c_hard_stroke,g_hard_stroke,g2_hard_stroke=extract_data(hard_stroke_file)
c_tickle,g_tickle,g2_tickle=extract_data(tickle_file)
c_hit,g_hit,g2_hit=extract_data(hit_file)

x_total=np.concatenate((g2_base[:,:,:,:], g2_air_stroke),axis=0)
x_total=np.concatenate((x_total,g2_light_stroke), axis=0)
x_total=np.concatenate((x_total,g2_hard_stroke), axis=0)
x_total=np.concatenate((x_total,g2_tickle), axis=0)
x_total=np.concatenate((x_total,g2_hit), axis=0)



#acquiring test data
g_base_t,g2_base_t=extract_baseline(baseline_file_t) #import x data for baseline
c_air_stroke_t,g_air_stroke_t,g2_air_stroke_t=extract_data(air_stroke_file_t) #import x data for air stroke
c_light_stroke_t,g_light_stroke_t,g2_light_stroke_t=extract_data(light_stroke_file_t) #import x data for light stroke
c_hard_stroke_t,g_hard_stroke_t,g2_hard_stroke_t=extract_data(hard_stroke_file_t)
c_tickle_t,g_tickle_t,g2_tickle_t=extract_data(tickle_file_t)
c_hit_t,g_hit_t,g2_hit_t=extract_data(hit_file_t)

x_total_t=np.concatenate((g2_base_t[:,:,:,:], g2_air_stroke_t),axis=0)
x_total_t=np.concatenate((x_total_t,g2_light_stroke_t), axis=0)
x_total_t=np.concatenate((x_total_t,g2_hard_stroke_t[50:,:,:,:]), axis=0)
x_total_t=np.concatenate((x_total_t,g2_tickle_t), axis=0)
x_total_t=np.concatenate((x_total_t,g2_hit_t), axis=0)



y_base=np.zeros(g2_base[:,:,:,:].shape[0]) #creating y labels for baseline
y_air_stroke=np.ones(g2_air_stroke.shape[0]) #creating y labels for stroke
y_light_stroke=2*np.ones(g2_light_stroke.shape[0]) #creating y labels for massage
y_hard_stroke=3*np.ones(g2_hard_stroke.shape[0])
y_tickle=4*np.ones(g2_tickle.shape[0])
y_hit=5*np.ones(g2_hit.shape[0])
y_total=np.append(y_base,y_air_stroke) #creating final y by adding base and stroke
y_total=np.append(y_total,y_light_stroke) #adding in massage
y_total=np.append(y_total,y_hard_stroke)
y_total=np.append(y_total,y_tickle)
y_total=np.append(y_total,y_hit)

y_all = to_categorical(y_total) #one hot encoding


#generate test labels
#
y_base_t=np.zeros(g2_base_t[:,:,:,:].shape[0]) #creating y labels for baseline
y_air_stroke_t=np.ones(g2_air_stroke_t.shape[0]) #creating y labels for stroke
y_light_stroke_t=2*np.ones(g2_light_stroke_t.shape[0]) #creating y labels for massage
y_hard_stroke_t=3*np.ones(g2_hard_stroke_t[50:,:,:,:].shape[0])
y_tickle_t=4*np.ones(g2_tickle_t.shape[0])
y_hit_t=5*np.ones(g2_hit_t.shape[0])
y_total_t=np.append(y_base_t,y_air_stroke_t) #creating final y by adding base and stroke
y_total_t=np.append(y_total_t,y_light_stroke_t) #adding in massage
y_total_t=np.append(y_total_t,y_hard_stroke_t)
y_total_t=np.append(y_total_t,y_tickle_t)
y_total_t=np.append(y_total_t,y_hit_t)

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

model.add(Dense(6, 
                activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#using validation data
#history=model.fit(X_train.reshape(X_train.shape[0],16,10,5,1), y_train,
#                  epochs=12,
#                  validation_data = (X_test.reshape(X_test.shape[0],16,10,5,1), y_test))
#

#using test data

history=model.fit(X_train.reshape(X_train.shape[0],16,10,5,1), y_train,
                  epochs=12,
                  validation_data = (x_total_t.reshape(x_total_t.shape[0],16,10,5,1), y_all_t))



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

pred=model.predict(x_total_t.reshape(x_total_t.shape[0],16,10,5,1))

y_true=np.array(tf.argmax(y_all_t,axis=1))
y_pred=np.array(tf.argmax(pred, axis=1))
cm=np.array(tf.math.confusion_matrix(y_true, y_pred))


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
#this creates the first frame of 5 data points
#----------------------------------------------------------
rt_frame_l=[]
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
        rt_frame_l.append(nn)

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
        
        rt_frame_l.append(nn)   
    
    
    rt_data=(np.array(rt_frame_l)-rt_baseline).reshape(1,16,10,5,1)/256
    
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
        print("OUCH!!")
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
