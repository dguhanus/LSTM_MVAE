# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:01:00 2019

@author: eledib
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:28:49 2019

@author: eledib
"""
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing

from scipy.io import loadmat


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42

dfdata = loadmat('mammography.mat')
nm = list(dfdata.items())
nm[4]
nm[4][1]
print(dfdata['X'].shape)
print(dfdata['y'].shape)
Data = dfdata['X']
scaler = preprocessing.StandardScaler()
Data = scaler.fit_transform(Data)
Label = dfdata['y']
X11 = np.hstack([Data,Label])
'''
 To verify whether columns are stacked properly or not 
X11[:,2]-Data[:,2]
Now naming the columns dynamically
'''

seed = 0

dy = X11[:,-1]
anamoly_dataset = X11[(dy==1)]
normal_dataset = X11[(dy==0)]
normal_dataset = normal_dataset[:4000]

ReducedData= np.concatenate((normal_dataset, anamoly_dataset))
np.random.seed(seed)
np.random.shuffle(ReducedData)

ano_frac = anamoly_dataset.shape[0]/(anamoly_dataset.shape[0]+normal_dataset.shape[0])
print('Anomaly percentage %f' % ano_frac)

train_x, test_x = train_test_split(ReducedData, test_size=0.2, random_state=RANDOM_SEED)

X_train = train_x[:,0:-1]
X_test = test_x[:,0:-1]

y_test = test_x[:,-1]

np.savetxt('MamoTrain-1-1.txt', X_train, delimiter=',', fmt='%d')
np.savetxt('MamoTest-1-1.txt', X_test, delimiter=',', fmt='%d')
np.savetxt('MamoTestLabel-1-1.txt', y_test, delimiter=',', fmt='%d')
