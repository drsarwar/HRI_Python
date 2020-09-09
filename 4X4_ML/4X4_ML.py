#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:16:49 2020

@author: saquib
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter 
from tkinter import filedialog as fd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import sys, serial


class read_serial(object):
    def __init__(self):
        self.complete_data=[[],[]]
        self.cap_data=[]


    def read_data(self,path):
        f=list(open(path,'r'))
        
        #this for loop is to get the number of taxels in the data
        for j in range(0,len(f)):
            taxel=int((f[j][1:5]),base=2)
            self.complete_data[0].append(taxel)
            temp=np.array(self.complete_data[0])
            self.num_taxel=len(np.unique(temp)) 
        
        #this for loop is to obtain all the capacitance data and then
        #to put the respective capacitance data in the respective list
        for i in range(0,len(f)):
            taxel=int((f[i][1:5]),base=2)
            self.complete_data[0].append(taxel)
            cap_value= float(f[i][7:13])
            self.complete_data[1].append(cap_value)
            if (len(self.cap_data)<self.num_taxel):
                self.cap_data.append([])
            else:
                self.cap_data[taxel].append(cap_value)
       
#adding multiple files to read in pressure shear and baseline data
                
path_base='/Users/saquib/Documents/Research/HRI/HRI_Python/4X4_ML/baseline.0'
#path_pressure='/Users/saquib/Documents/Research/HRI/HRI_Python/4X4_ML/pressure.0'
#path_shear='/Users/saquib/Desktop/ML/shear.0'
path_prox='/Users/saquib/Documents/Research/HRI/HRI_Python/4X4_ML/proximity.0'
path_touch='/Users/saquib/Documents/Research/HRI/HRI_Python/4X4_ML/touch.0'

X_base=read_serial()
#X_pressure=read_serial()
#X_shear=read_serial()
X_prox=read_serial()
X_touch=read_serial()

X_base.read_data(path_base)
#X_pressure.read_data(path_pressure)
#X_shear.read_data(path_shear)
X_prox.read_data(path_prox)
X_touch.read_data(path_touch)

base_data=np.array(np.transpose(X_base.cap_data))
#pressure_data=np.array(np.transpose(X_pressure.cap_data))
#shear_data=np.array(np.transpose(X_shear.cap_data))
prox_data=np.array(np.transpose(X_prox.cap_data))
touch_data=np.array(np.transpose(X_touch.cap_data))

for j in range(16):
    plt.plot(prox_data[:,j])



Y_base=np.empty(len(base_data),dtype='int')
for i in range(0,len(base_data)):
    Y_base[i]=0
#Y_pressure=np.empty(len(pressure_data),dtype='int')
#for i in range(0,len(pressure_data)):
#    Y_pressure[i]=1
# Y_shear=np.empty(len(shear_data),dtype='int')
# for i in range(0,len(shear_data)):
#     Y_shear[i]=2
Y_prox=np.empty(len(prox_data),dtype='int')
for i in range(0,len(prox_data)):
    Y_prox[i]=2
Y_touch=np.empty(len(touch_data),dtype='int')
for i in range(0,len(touch_data)):
    Y_touch[i]=3
    
    
#X=np.append(base_data,pressure_data,axis=0)
# X=np.append(X,shear_data,axis=0)
X=np.append(base_data,prox_data,axis=0) #use this if there is no pressure data
#X=np.append(X,prox_data,axis=0) #use this if there is pressure data
X=np.append(X,touch_data,axis=0)

#y=np.append(Y_base,Y_pressure,axis=0)
# y=np.append(y,Y_shear,axis=0)
y=np.append(Y_base,Y_prox,axis=0) #use this if there is no pressure data
#y=np.append(y,Y_prox,axis=0) #use this if there is pressure data
y=np.append(y,Y_touch,axis=0)

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.3, stratify=y)


##---------------------------------
# #grid search for best parameters
##---------------------------------

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc = SVC(C=100, kernel='rbf', gamma = 100, 
          class_weight='balanced')

grid={'C':[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2],
      'gamma':[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2]}


search = GridSearchCV(estimator=svc,
                      param_grid=grid,
                      scoring='accuracy',
                      cv=5)
search.fit(X_train,y_train)

clf = SVC(C=10, kernel='rbf', gamma = 100, 
          class_weight='balanced')
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test) #using clf.score
#acc=accuracy_score(y_test, clf.predict(X_test)) #using accuracy_score from sklearn

#----------------------------
#Random Forest classifier
#----------------------------

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier

# clf = RandomForestClassifier(n_estimators = 10, random_state = 1, class_weight='balanced')





#----------------------------
#Adaboost using default tree estimator
#----------------------------



# clf = AdaBoostClassifier(n_estimators=20,learning_rate=0.1,random_state=1)

#------------------------
#Adaboost using SVM estimator
#------------------------

# # Import Support Vector Classifier
# from sklearn.svm import SVC
# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# svc=SVC(probability=True, kernel='linear')

# # Create adaboost classifer object
# clf =AdaBoostClassifier(n_estimators=10, base_estimator=svc,learning_rate=0.5)


# Train the model on training data


# clf.fit(X_train,y_train)
# a1=accuracy_score(y_test, clf.predict(X_test))

#---------------------------------
#real time recognition of stimulus
#---------------------------------



import time

strPort = '/dev/cu.usbmodem143101'
ser = serial.Serial(strPort, baudrate=9600)
cnt=0
stat=np.zeros((16),dtype='float')


while True:
    # start = time.process_time()
    line = ser.readline()
    data=str(line)
    if (data[2]=='(') and (data[7]==')'):
        indx=int(data[3:7],2)
        cnt=cnt+1
        if indx==0:
            stat[0]=float(data[8:13])
        elif indx==1:
            stat[1]=float(data[8:13])
        elif indx==2:
            stat[2]=float(data[8:13])
        elif indx==3:
            stat[3]=float(data[8:13])
        elif indx==4:
            stat[4]=float(data[8:13])
        elif indx==5:
            stat[5]=float(data[8:13])
        elif indx==6:
            stat[6]=float(data[8:13])
        elif indx==7:
            stat[7]=float(data[8:13])
        elif indx==8:
            stat[8]=float(data[8:13])
        elif indx==9:
            stat[9]=float(data[8:13])
        elif indx==10:
            stat[10]=float(data[8:13])
        elif indx==11:
            stat[11]=float(data[8:13])
        elif indx==12:
            stat[12]=float(data[8:13])
        elif indx==13:
            stat[13]=float(data[8:13])
        elif indx==14:
            stat[14]=float(data[8:13])
        elif indx==15:
            stat[15]=float(data[8:13])
            
        if cnt==15:
            #print(stat)
            if clf.predict(stat.reshape(1,16))==0:
                print('baseline')
#            elif clf.predict(stat.reshape(1,16))==1:
#                print('pressure')
            elif clf.predict(stat.reshape(1,16))==2:
                print('proximity')
            # elif clf.predict(stat.reshape(1,16))==3:
            #     print('proximity')
            else:
                print('touch')
            #print(clf.predict(stat.reshape(1,4)))
            # print((time.process_time() - start)*1000)
            cnt=0
    else:
            continue
  


#for single data file case
                
pa='/Users/saquib/Desktop/screenlog.0'
X_single=read_serial()
X_single.read_data(pa)

single_data=np.array(np.transpose(X_single.cap_data))

for i in range(16):
    plt.plot(single_data[:,i])

c_data_np=np.array(np.transpose(X_single.cap_data))
time=np.linspace(0,len(c_data_np)-1,len(c_data_np))
plt.plot(c_data_np[:,0],'r',c_data_np[:,1],'b',c_data_np[:,2],'g',c_data_np[:,3],'y')

#calculating the mean of the four pads using zip
zip_along_row=zip(np.transpose(np.array(X_single.cap_data)))
list_zip=list(zip_along_row)
mean_data_points=[np.mean(list_zip[i]) for i in range(0,len(list_zip))]
#calculating using zip ends here

# #----------------------------
# #creating the feature space
# #----------------------------

# #adding the mean along col 4
# feat=np.append(c_data_np, np.reshape(mean_data_points,(-1,1)),axis=1)

# #adding the difference between 0 and 2
# diff_0_2=np.subtract(c_data_np[:,2],c_data_np[:,0])
# feat=np.append(feat, np.reshape(diff_0_2, (-1,1)), axis=1)

# #adding the difference between 1 and 3
# diff_1_3=np.subtract(c_data_np[:,3],c_data_np[:,1])
# feat=np.append(feat, np.reshape(diff_1_3, (-1,1)), axis=1)

