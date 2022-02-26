# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:48:21 2019

@author: ADMIN
"""

"""
=====================================
Blind source separation using FastICA
=====================================

An example of estimating sources from noisy data.

:ref:`ICA` is used to estimate sources given noisy measurements.
Imagine 3 instruments playing simultaneously and 3 microphones
recording the mixed signals. ICA is used to recover the sources
ie. what is played by each instrument. Importantly, PCA fails
at recovering our `instruments` since the related signals reflect
non-Gaussian processes.

"""
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA


# #############################################################################
# Generate sample data
#np.random.seed(0)
#n_samples = 2000
#time = np.linspace(0, 8, n_samples)
#
#s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
#s2 = np.sign(np.sin(0 * time))  # Signal 2 : square signal
#s3 = signal.sawtooth(0 * np.pi * time)  # Signal 3: saw tooth signal
#
#S = np.c_[s1, s2, s3]
#S += 0.2 * np.random.normal(size=S.shape)  # Add noise
#
#S /= S.std(axis=0)  # Standardize data
## Mix data
#A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
#X = np.dot(S, A.T)  # Generate observations

import pywt

def Meig(dfdata):
    data = dfdata
    waveletname = 'db4'
    cA_1, cD_1 = pywt.dwt(data, waveletname)
    cA_2, cD_2 = pywt.dwt(cA_1, waveletname)
    cA_3, cD_3 = pywt.dwt(cA_2, waveletname)
    tD_3 = np.pad(cD_3, (0, cD_1.size - cD_3.size), mode='constant', constant_values=0)
    tD_2 = np.pad(cD_2, (0, cD_1.size - cD_2.size), mode='constant', constant_values=0)
    tA_3 = np.pad(cA_3, (0, cD_1.size - cA_3.size), mode='constant', constant_values=0)
    cD_3 = tD_3
    cD_2 = tD_2
    cA_3 = tA_3
    X= np.c_[cD_1.T, cD_2.T, cD_3.T, cA_3.T]
    # Compute ICA
    #ica = FastICA(n_components=4)
    #S_ = ica.fit_transform(X)  # Reconstruct signals
    #A_ = ica.mixing_  # Get estimated mixing matrix
    # We can `prove` that the ICA model applies by reverting the unmixing.
    #assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
    # For comparison, compute PCA
    #pca = PCA(n_components=3)
    #H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
    # #############################################################################
    # Plot results
    #plt.figure()
    #models = [X, S_, H]
    #names = ['Observations (mixed signal)', 'ICA recovered signals', 'PCA recovered signals']
    #colors = ['red', 'steelblue', 'orange']
    X_cov = np.cov(X.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(X_cov)
    return eig_val_cov[:2]

def MultiVarPCA(dfdata):
    X = dfdata
    X_cov = np.cov(X.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(X_cov)
    return eig_val_cov[:2]

def Meig_Zero(dfdata):
    data = dfdata
    waveletname = 'db4'
    cA_1, cD_1 = pywt.dwt(data, waveletname)
    cA_2, cD_2 = pywt.dwt(cA_1, waveletname)
    cA_3, cD_3 = pywt.dwt(cA_2, waveletname)
    cA_4, cD_4 = pywt.dwt(cA_3, waveletname)
    tD_4 = np.pad(cD_4, (0, cD_1.size - cD_4.size), mode='constant', constant_values=0)
    tD_3 = np.pad(cD_3, (0, cD_1.size - cD_3.size), mode='constant', constant_values=0)
    tD_2 = np.pad(cD_2, (0, cD_1.size - cD_2.size), mode='constant', constant_values=0)
    tA_3 = np.pad(cA_3, (0, cD_1.size - cA_3.size), mode='constant', constant_values=0)
    cD_3 = tD_3
    cD_2 = tD_2
    cD_4 = tD_4
    X= np.c_[cD_1.T, cD_2.T, cD_3.T, cD_4.T]
    # Compute ICA
    #ica = FastICA(n_components=4)
    #S_ = ica.fit_transform(X)  # Reconstruct signals
    #A_ = ica.mixing_  # Get estimated mixing matrix
    # We can `prove` that the ICA model applies by reverting the unmixing.
    #assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
    # For comparison, compute PCA
    #pca = PCA(n_components=3)
    #H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
    # #############################################################################
    # Plot results
    #plt.figure()
    #models = [X, S_, H]
    #names = ['Observations (mixed signal)', 'ICA recovered signals', 'PCA recovered signals']
    #colors = ['red', 'steelblue', 'orange']
    X_cov = np.cov(X.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(X_cov)
    return eig_val_cov[:2]