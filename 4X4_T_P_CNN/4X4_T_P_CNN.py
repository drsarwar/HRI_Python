#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:26:04 2020

@author: saquib
"""

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
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.utils import plot_model
import sys, serial


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

def crop(all_data):
    
    data_delta=np.zeros(shape=all_data.shape)
    for j in range(all_data.shape[1]-1):
        data_delta[:,j-1]=all_data[:,j-1]-np.mean(all_data[:1,j-1])
        
    data_ac=np.array([])
    data_crop=np.zeros(shape=(1,17))
    
    for k in range(data_delta.shape[0]):
        for l in range(data_delta.shape[1]):
           # if total_data_delta[k,l]>0.05 and np.count_nonzero(data_ac == k)==0:
            if  data_delta[k,l]>0.05:
                data_crop=np.vstack((data_crop,all_data[k,:].reshape(1,-1)))
                data_ac=np.append(data_ac,k)
                break
    return data_crop[1:,:]



baseline='/Users/saquib/Desktop/4X4_CNN_T_P/baseline.0'
proximity='/Users/saquib/Desktop/4X4_CNN_T_P/proximity.0'
touch='/Users/saquib/Desktop/4X4_CNN_T_P/touch.0'
proximity_test='/Users/saquib/Desktop/4X4_CNN_T_P/proximity_test.0'
touch_test='/Users/saquib/Desktop/4X4_CNN_T_P/touch_test.0'


X_base=read_serial()
X_base.read_data(baseline)
data_base=np.array(X_base.cap_data)

X_proximity=read_serial()
X_proximity.read_data(proximity)
data_proximity=np.array(X_proximity.cap_data)

X_touch=read_serial()
X_touch.read_data(touch)
data_touch=np.array(X_touch.cap_data)


for i in range(16):
    plt.plot(data_base[i,:])


data=np.append(data_base,data_proximity,axis=1)
data=np.append(data,data_touch, axis=1)

y_base=np.zeros(data_base.shape[1])
y_proximity=1*np.ones(data_proximity.shape[1])
y_touch=2*np.ones(data_touch.shape[1])

y=np.append(y_base, y_proximity)
y=np.append(y, y_touch)

data_t=np.transpose(data)

total_data = np.hstack((data_t,(y.reshape(-1,1))))

#for i in range(16):
#    plt.plot(data_delta[:,i])

#---------
#test code to check the base_mean is correct
#--------------
    
#base_mean=np.zeros(shape=data_t.shape)
#base_cap_t=base_cap.reshape(1,-1)
#for s in range(data_t.shape[0]):
#    base_mean[s]=base_cap_t
#
#
#---------------------------

data_delta=np.zeros(shape=data_t.shape)
for j in range(data_t.shape[1]):
    data_delta[:,j]=data_t[:,j]-np.mean(data_t[:10,j])
    
#-----------------------------------------------#
#this segment is for test data only
#-----------------------------------------------#

X_proximity_test=read_serial()
X_proximity_test.read_data(proximity_test)
data_proximity_test=np.array(X_proximity_test.cap_data)

y_proximity_test=np.ones(data_proximity_test.shape[1])

data_proximity_test_t=np.transpose(data_proximity_test)
total_proximity_test=np.hstack((data_proximity_test_t,(y_proximity_test.reshape(-1,1))))


X_touch_test=read_serial()
X_touch_test.read_data(touch_test)
data_touch_test=np.array(X_touch_test.cap_data)

y_touch_test=2*np.ones(data_touch_test.shape[1])

data_touch_test_t=np.transpose(data_touch_test)
total_touch_test=np.hstack((data_touch_test_t,(y_touch_test.reshape(-1,1))))

total_test_data=np.vstack((total_proximity_test,total_touch_test))


X_test_data=np.append(data_proximity_test,data_touch_test,axis=1)
X_test_data=np.transpose(X_test_data)



test_data_delta=np.zeros(shape=X_test_data.shape)
for j in range(X_test_data.shape[1]):
    test_data_delta[:,j]=X_test_data[:,j]-np.mean(data_t[:10,j])
    

y_proximity_test=1*np.ones(data_proximity_test.shape[1])
y_touch_test=2*np.ones(data_touch_test.shape[1])

y_test=np.append(y_proximity_test, y_touch_test)
y_test_cat=to_categorical(y_test)

#------------------------------------------------#

#total_data_delta = np.hstack((data_delta,(y.reshape(-1,1))))

#plt.figure(2)
#plt.plot(data[15,:])        
#for i in range(16):
#    plt.plot(data_delta[:,i],'o')    


#to make dataset we only want to keep the parts of the 
#data that correspond to a stimulus
#succesful crop for any stimulus over 1% change

#
#
#data_ac=np.array([])
#data_crop=np.zeros(shape=(1,17))
#
#for k in range(total_data_delta.shape[0]):
#    for l in range(total_data_delta.shape[1]-1):
#       # if total_data_delta[k,l]>0.05 and np.count_nonzero(data_ac == k)==0:
#        if total_data_delta[k,l]>0.05:
#            data_crop=np.vstack((data_crop,total_data[k,:].reshape(1,-1)))
#            data_ac=np.append(data_ac,k)
#            break
#            


#added this part to keep the discarded data
#
#data_ac_base=np.array([])
#data_discard_base=np.zeros(shape=(1,17))
#
#for m in range(total_data_delta.shape[0]):
#    if np.count_nonzero(data_ac == m)==0:
#        data_discard_base=np.vstack((data_discard_base,total_data[m,:].reshape(1,-1)))
#        data_ac_base=np.append(data_ac_base,m)
#data_discard_base[:,16]=0

#ends here
        

#add in the baseline

##
##X_with_base=np.vstack((data_discard_base,data_crop)) #to keep discarded data
##X_with_base=np.delete(X_with_base,0,0) #this is for data with discard
##X_with_base=np.delete(X_with_base,1650,0) #this is for data with discard
##
#X_with_base=np.vstack((total_data[0:1058,:],data_crop)) #this is for regular
#X_with_base=np.delete(X_with_base,1058,0) #this is for data w/o discard
#
##X=X_with_base[500:,:16]
#
#X=X_with_base[:,:16]
#X_norm=(X-np.min(X))/(np.max(X)-np.min(X))
##y=X_with_base[500:,16]
#y=X_with_base[:,16]
#
#y_all = to_categorical(y)


#-------------------
#this part for touch/prox
#-------------------

#X=total_data[:,:16]

X=data_delta
y=total_data[:,16]

y_all = to_categorical(y)

#--------------------------------#

#adding test data of M

#X_with_M=np.vstack((X,test_data[:,:16]))
#y_M_cat=to_categorical(test_data[:,16])
#y_with_M=np.vstack((y_all,y_M_cat))
#
#
#
#
#X_train, X_test, y_train, y_test = train_test_split(
#         X_with_M, y_with_M, test_size=0.3, random_state=1)
#

#--------------------------------#



X_train, X_test, y_train, y_test = train_test_split(
         X, y_all, test_size=0.3, random_state=1)
#
#for i in range(16):
#    plt.plot(X[:,i])    
#    


data2D_train=np.zeros(shape=(X_train.shape[0],4,4))
#mapping for 4X4 board 3mm with arduino mega
for i in range(X_train.shape[0]):
    data2D_train[i,0,0]=X_train[i,5]
    data2D_train[i,0,1]=X_train[i,4]
    data2D_train[i,0,2]=X_train[i,7]
    data2D_train[i,0,3]=X_train[i,6]
    data2D_train[i,1,0]=X_train[i,1]
    data2D_train[i,1,1]=X_train[i,0]
    data2D_train[i,1,2]=X_train[i,3]
    data2D_train[i,1,3]=X_train[i,2]
    data2D_train[i,2,0]=X_train[i,13]
    data2D_train[i,2,1]=X_train[i,12]
    data2D_train[i,2,2]=X_train[i,15]
    data2D_train[i,2,3]=X_train[i,14]
    data2D_train[i,3,0]=X_train[i,9]
    data2D_train[i,3,1]=X_train[i,8]
    data2D_train[i,3,2]=X_train[i,11]
    data2D_train[i,3,3]=X_train[i,10]




data2D_test=np.zeros(shape=(X_test.shape[0],4,4))
#mapping for 4X4 board 3mm with arduino mega
for i in range(X_test.shape[0]):
    data2D_test[i,0,0]=X_test[i,5]
    data2D_test[i,0,1]=X_test[i,4]
    data2D_test[i,0,2]=X_test[i,7]
    data2D_test[i,0,3]=X_test[i,6]
    data2D_test[i,1,0]=X_test[i,1]
    data2D_test[i,1,1]=X_test[i,0]
    data2D_test[i,1,2]=X_test[i,3]
    data2D_test[i,1,3]=X_test[i,2]
    data2D_test[i,2,0]=X_test[i,13]
    data2D_test[i,2,1]=X_test[i,12]
    data2D_test[i,2,2]=X_test[i,15]
    data2D_test[i,2,3]=X_test[i,14]
    data2D_test[i,3,0]=X_test[i,9]
    data2D_test[i,3,1]=X_test[i,8]
    data2D_test[i,3,2]=X_test[i,11]
    data2D_test[i,3,3]=X_test[i,10]
    
    

    
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

#
#


#data2D_delta=np.zeros(shape=(data_delta.shape[0],4,4))
##mapping for 4X4 board 3mm with arduino mega
#for i in range(data_delta.shape[0]):
#    data2D_delta[i,0,0]=data_delta[i,5]
#    data2D_delta[i,0,1]=data_delta[i,4]
#    data2D_delta[i,0,2]=data_delta[i,7]
#    data2D_delta[i,0,3]=data_delta[i,6]
#    data2D_delta[i,1,0]=data_delta[i,1]
#    data2D_delta[i,1,1]=data_delta[i,0]
#    data2D_delta[i,1,2]=data_delta[i,3]
#    data2D_delta[i,1,3]=data_delta[i,2]
#    data2D_delta[i,2,0]=data_delta[i,13]
#    data2D_delta[i,2,1]=data_delta[i,12]
#    data2D_delta[i,2,2]=data_delta[i,15]
#    data2D_delta[i,2,3]=data_delta[i,14]
#    data2D_delta[i,3,0]=data_delta[i,9]
#    data2D_delta[i,3,1]=data_delta[i,8]
#    data2D_delta[i,3,2]=data_delta[i,11]
#    data2D_delta[i,3,3]=data_delta[i,10]
#    
#    

data2D_delta_test=np.zeros(shape=(test_data_delta.shape[0],4,4))
#mapping for 4X4 board 3mm with arduino mega
for i in range(test_data_delta.shape[0]):
    data2D_delta_test[i,0,0]=test_data_delta[i,5]
    data2D_delta_test[i,0,1]=test_data_delta[i,4]
    data2D_delta_test[i,0,2]=test_data_delta[i,7]
    data2D_delta_test[i,0,3]=test_data_delta[i,6]
    data2D_delta_test[i,1,0]=test_data_delta[i,1]
    data2D_delta_test[i,1,1]=test_data_delta[i,0]
    data2D_delta_test[i,1,2]=test_data_delta[i,3]
    data2D_delta_test[i,1,3]=test_data_delta[i,2]
    data2D_delta_test[i,2,0]=test_data_delta[i,13]
    data2D_delta_test[i,2,1]=test_data_delta[i,12]
    data2D_delta_test[i,2,2]=test_data_delta[i,15]
    data2D_delta_test[i,2,3]=test_data_delta[i,14]
    data2D_delta_test[i,3,0]=test_data_delta[i,9]
    data2D_delta_test[i,3,1]=test_data_delta[i,8]
    data2D_delta_test[i,3,2]=test_data_delta[i,11]
    data2D_delta_test[i,3,3]=test_data_delta[i,10]
    
    
#
#
#
#data2D_crop=np.zeros(shape=(4,4,data_t.shape[0]))
##mapping for 4X4 board 3mm with arduino mega
#for i in range(data_crop.shape[0]):
#    data2D_crop[0,0,i]=data_crop[i,5]
#    data2D_crop[0,1,i]=data_crop[i,4]
#    data2D_crop[0,2,i]=data_crop[i,7]
#    data2D_crop[0,3,i]=data_crop[i,6]
#    data2D_crop[1,0,i]=data_crop[i,1]
#    data2D_crop[1,1,i]=data_crop[i,0]
#    data2D_crop[1,2,i]=data_crop[i,3]
#    data2D_crop[1,3,i]=data_crop[i,2]
#    data2D_crop[2,0,i]=data_crop[i,13]
#    data2D_crop[2,1,i]=data_crop[i,12]
#    data2D_crop[2,2,i]=data_crop[i,15]
#    data2D_crop[2,3,i]=data_crop[i,14]
#    data2D_crop[3,0,i]=data_crop[i,9]
#    data2D_crop[3,1,i]=data_crop[i,8]
#    data2D_crop[3,2,i]=data_crop[i,11]
#    data2D_crop[3,3,i]=data_crop[i,10]



def animate(i):
    data_a = data2D_delta_test[i,:,:] #select data range
#    plt.imshow(data_a, vmin=-0.02,vmax=0.56, cmap='gray')
    plt.imshow(data_a, vmin=-0.05, vmax=0.0002, cmap='gray')

fig=plt.figure(1)
ani = animation.FuncAnimation(fig, animate, frames=data2D_train.shape[0], interval=10, repeat=True)
#ani.save('/Users/saquib/Desktop/4X4_CNN/ani.mp4', fps=3,extra_args=['-vcodec', 'libx264'])
plt.show()
    



model=Sequential()

model.add(Conv2D(20,
                 kernel_size=2,
                 activation='relu',
                 input_shape=(4, 4, 1),
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
model.add(Dense(32, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l=0.01)))
model.add(Dense(16, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l=0.01)))
model.add(Dense(3, 
                activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#history=model.fit(data2D_train.reshape(data2D_train.shape[0],4,4,1), y_train,
#                  epochs=5,
#                  validation_data = (data2D_test.reshape(data2D_test.shape[0],4,4,1), y_test))

history=model.fit(data2D_train.reshape(data2D_train.shape[0],4,4,1), y_train,
                  epochs=12,
                  validation_data = (data2D_delta_test.reshape(data2D_delta_test.shape[0],4,4,1), y_test_cat))


data2D_CNN_shape=data2D_test.reshape(data2D_test.shape[0],4,4,1)




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



scores=model.evaluate(data2D_train.reshape(data2D_train.shape[0],4,4,1),y_train,verbose=1)


pp=model.predict(data2D_delta_test.reshape(data2D_delta_test.shape[0],4,4,1))

#-------------------------------------------------------#

#This segment is for real time recognition

#-------------------------------------------------------#


strPort = '/dev/cu.usbmodem143101'
ser = serial.Serial(strPort, baudrate=9600)
cnt=0
stat=np.zeros((16),dtype='float')

base_cap=np.mean(data_t[:100,:],axis=0)


fig,ax = plt.subplots(1,1)
image = data2D_delta_test[0,:,:]
im = ax.imshow(image,vmin=-0.05, vmax=0.0002,cmap='gray')

while True:
    # start = time.process_time()
    line = ser.readline()
    data=str(line)
    if (data[2]=='(') and (data[7]==')'):
        indx=int(data[3:7],2)
        cnt=cnt+1
     
#        if indx==0:
#            stat[0]=float(data[8:13])
#        elif indx==1:
#            stat[1]=float(data[8:13])
#        elif indx==2:
#            stat[2]=float(data[8:13])
#        elif indx==3:
#            stat[3]=float(data[8:13])
#        elif indx==4:
#            stat[4]=float(data[8:13])
#        elif indx==5:
#            stat[5]=float(data[8:13])
#        elif indx==6:
#            stat[6]=float(data[8:13])
#        elif indx==7:
#            stat[7]=float(data[8:13])
#        elif indx==8:
#            stat[8]=float(data[8:13])
#        elif indx==9:
#            stat[9]=float(data[8:13])
#        elif indx==10:
#            stat[10]=float(data[8:13])
#        elif indx==11:
#            stat[11]=float(data[8:13])
#        elif indx==12:
#            stat[12]=float(data[8:13])
#        elif indx==13:
#            stat[13]=float(data[8:13])
#        elif indx==14:
#            stat[14]=float(data[8:13])
#        elif indx==15:
#            stat[15]=float(data[8:13])
#            
        if indx==0:
            stat[0]=float(data[8:13])-base_cap[0]
        elif indx==1:
            stat[1]=float(data[8:13])-base_cap[1]
        elif indx==2:
            stat[2]=float(data[8:13])-base_cap[2]
        elif indx==3:
            stat[3]=float(data[8:13])-base_cap[3]
        elif indx==4:
            stat[4]=float(data[8:13])-base_cap[4]
        elif indx==5:
            stat[5]=float(data[8:13])-base_cap[5]
        elif indx==6:
            stat[6]=float(data[8:13])-base_cap[6]
        elif indx==7:
            stat[7]=float(data[8:13])-base_cap[7]
        elif indx==8:
            stat[8]=float(data[8:13])-base_cap[8]
        elif indx==9:
            stat[9]=float(data[8:13])-base_cap[9]
        elif indx==10:
            stat[10]=float(data[8:13])-base_cap[10]
        elif indx==11:
            stat[11]=float(data[8:13])-base_cap[11]
        elif indx==12:
            stat[12]=float(data[8:13])-base_cap[12]
        elif indx==13:
            stat[13]=float(data[8:13])-base_cap[13]
        elif indx==14:
            stat[14]=float(data[8:13])-base_cap[14]
        elif indx==15:
            stat[15]=float(data[8:13])-base_cap[15]
            
        if cnt==15:
            #print(stat)
            
            #------------------------#
            #using baseline correction
#            
#                        
#            stat_delta=np.zeros(shape=stat.shape)
#            for j in range(stat.shape[0]):
#                stat_delta[j]=stat[j]-np.mean(data_t[:10,j])
#    
#            stat=stat_delta
            #----------------------#
            
            signal2D=np.zeros(shape=(4,4))
            #mapping for 4X4 board 3mm with arduino mega
            
            signal2D[0,0]=stat[5]
            signal2D[0,1]=stat[4]
            signal2D[0,2]=stat[7]
            signal2D[0,3]=stat[6]
            signal2D[1,0]=stat[1]
            signal2D[1,1]=stat[0]
            signal2D[1,2]=stat[3]
            signal2D[1,3]=stat[2]
            signal2D[2,0]=stat[13]
            signal2D[2,1]=stat[12]
            signal2D[2,2]=stat[15]
            signal2D[2,3]=stat[14]
            signal2D[3,0]=stat[9]
            signal2D[3,1]=stat[8]
            signal2D[3,2]=stat[11]
            signal2D[3,3]=stat[10]

            if np.argmax(model.predict(signal2D.reshape(1,4,4,1)),axis=1)==0:
                print('baseline')
            elif np.argmax(model.predict(signal2D.reshape(1,4,4,1)),axis=1)==1:
                print('proximity')
            elif np.argmax(model.predict(signal2D.reshape(1,4,4,1)),axis=1)==2:
                print('touch')
            # elif clf.predict(stat.reshape(1,16))==3:
            #     print('proximity')
            else:
                print('unrecognized')
            #print(clf.predict(stat.reshape(1,4)))
            # print((time.process_time() - start)*1000)
            image = signal2D
            im.set_data(image)
            fig.canvas.draw_idle()
            plt.pause(0.001)
            cnt=0
    else:
            continue
  

