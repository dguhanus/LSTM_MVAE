# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:16:27 2019

@author: eledib
"""
import numpy as np
import scipy as sp
import pandas as pd

from scipy.stats import chi2

# define a function
def print_factors(x):
   # This function takes a number and prints the factors

   print("The factors of",x,"are:")
   for i in range(1, x + 1):
       if x % i == 0:
           print(i)

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    Mohala = mahal.diagonal()
    Mohala = Mohala.reshape(-1,1)
    PcaMahaDst = np.concatenate((x, Mohala),axis=1)
    df_pca = pd.DataFrame(PcaMahaDst, columns=['pca1', 'pca2', 'MahaDist'])
    chi2.ppf((1-0.05), df=2)
    df_pca['p_value'] = 1 - chi2.cdf(df_pca['MahaDist'], 2)
    # Extreme values with a significance level of 0.01
    outlier_pca_05=df_pca.loc[df_pca.p_value < 0.05]
    normal_pca_05=df_pca.loc[df_pca.p_value > 0.05]
    normal_data_pca = np.array([normal_pca_05.pca1, normal_pca_05.pca2]).T
    cova = np.cov(normal_data_pca.T)
    return cova

def get_data(data):
    timesteps = 5   
    dataX = []
    for i in range(len(data) - timesteps + 1):
        x = data[i:(i+timesteps)]
        dataX.append(x)
    return np.array(dataX)

def computeGCD(x, y): 
  
    if x > y: 
        small = y 
    else: 
        small = x 
    for i in range(1, small+1): 
        if((x % i == 0) and (y % i == 0)): 
            gcd = i 
              
    return gcd

