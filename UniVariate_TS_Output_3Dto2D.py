# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:32:32 2019
THis is for restructuring output data of 3D into 2D
@author: eledib
"""

import numpy as np
from numpy import array

#def get_data(data):
#    # read data from file
#    
#    timesteps = 4
#    dataX = []
#    for i in range(len(data) - timesteps+1):
#        x = data[i:(i+timesteps)]
#        dataX.append(x)
#    return np.array(dataX)
#
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
## choose a number of time steps
#X_3D = get_data(raw_seq)

#print(X_3D)
#X_3D.shape

def Convert_2D(data):
    X_temp = []
    for i in range(data.shape[0]-1):
        X_temp.append(data[i,0])

    for j in range(data.shape[1]):
        X_temp.append(data[data.shape[0]-1,j])
    return np.array(X_temp)

#X_2D = Convert_2D(X_3D)
#print("Printing after restructuring the output", X_2D)
