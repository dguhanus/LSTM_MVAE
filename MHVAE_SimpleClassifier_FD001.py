# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 23:08:09 2020
https://github.com/TobiasGlaubach/python-ml-turbofan
https://www.kaggle.com/billstuart/predictive-maintenance-ml-iiot
https://www.rrighart.com/blog-gatu/sensor-time-series-of-aircraft-engines
@author: eledib
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV


from LoadData import load_data
import PlottingHelpers
import ProcessingHelpers

importlib.reload(ProcessingHelpers) # while still working on than fun
importlib.reload(PlottingHelpers) # while still working on than fun

sns.set() 

dirname = os.getcwd()
pth = os.path.join(dirname, 'CMAPSSData')

print('loading data...')
dc = load_data(pth)
print('done')

# get the first data set training data
df = dc['FD_001']['df_train'].copy()
'''
Make a Column for the RUL data
According to the data description document the data set contains multiple units, each unit 
starts 
at a certain degradation point and the measurement data ends closely before the unit was 
decommissioned of broke.

Therefore assume, that for the last measurement time that is available for a unit the units 
RUL=0 (stopped measuring just before machine broke)
'''
# get the time of the last available measurement for each unit
mapper = {}
for unit_nr in df['unit_nr'].unique():
    mapper[unit_nr] = df['time'].loc[df['unit_nr'] == unit_nr].max()

''' calculate RUL = time.max() - time_now for each unit'''
df['RUL'] = df['unit_nr'].apply(lambda nr: mapper[nr]) - df['time']
'''
Map the problem to a binary classification problem by mapping the RUL column to a binary 
colum by predicting a breakdown within the next 10 step
'''
df['FAILURE_NEAR'] = df['RUL'] < 10.0
'''
Drop the nan columns and rows
'''
cols_nan = df.columns[df.isna().any()].tolist()
print('Columns with all nan: \n' + str(cols_nan) + '\n')

cols_const = [ col for col in df.columns if len(df[col].unique()) <= 2 and col != 'FAILURE_NEAR' ]
print('Columns with all const values: \n' + str(cols_const) + '\n')

df = df.drop(columns=cols_const + cols_nan)
'''
Normalize the dataset as shown in the Explorative_analysis notebook
'''
df_start = df.loc[df['time'] < 10.0].copy()
cols_non_data = [col for col in df.columns if not col.startswith('sens')]
bias = df_start.mean()
scale = df_start.var()

bias[cols_non_data] = 0.0
scale[cols_non_data] = 1.0

df_n = (df - bias) / scale
#df_n = df.copy()
'''
take out a certain percentage of units from the training data set for testing later, (additionally to the classic validation methods)
'''
units = df_n['unit_nr'].unique()
n_units = len(df_n['unit_nr'].unique())

units_test = random.sample(list(units), int(n_units * 0.2))
units_train = [nr for nr in units if nr not in units_test]

df_n_test = df_n.loc[df_n['unit_nr'].apply( lambda x: x in units_test )].copy()
df_n_train = df_n.loc[df_n['unit_nr'].apply( lambda x: x in units_train )].copy()
df_n_train.describe()

cols_features = [c for c in df_n_train.columns if c.startswith('o') or c.startswith('s')]

PlottingHelpers.plot_imshow(df_n_train, resample=False)

c = cols_features + ['FAILURE_NEAR']
t = pd.DataFrame(df_n_test[c].values, columns=c).plot(subplots=True, figsize=(15, 15))

c = cols_features + ['FAILURE_NEAR']
t = pd.DataFrame(df_n_train[c].values, columns=c).plot(subplots=True, figsize=(15, 15))

'''
Do a simple support vector machine based classification on all training data
define a helper function for the simple fitting through a support vector machine
'''
from DeepVAEMSE import *

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

X_train = df_n_train[cols_features].values
X_train = scaler.fit_transform(X_train)


y_train = df_n_train['FAILURE_NEAR'].values > 0
  
X_test = df_n_test[cols_features].values
X_test = scaler.transform(X_test)

y_test = df_n_test['FAILURE_NEAR'].values

from Gcd_Factorise import get_data, mahalanobis

x_pca = np.zeros((X_train.shape[0], 2))
xcov = np.cov(X_train.T)
eigen_values, eigen_vectors = np.linalg.eig(xcov)
egn_vectors = eigen_vectors[:,:2]
x_pca = np.matmul(X_train, egn_vectors)

from Gcd_Factorise import get_data, mahalanobis
covar = mahalanobis(x = x_pca, data = x_pca)

X_normalized = preprocessing.normalize(covar, norm='l2')
std_dev_feed = X_normalized

if X_train.shape[0]%2 == 1:
    X_train=X_train[:-1,:]
if X_test.shape[0]%2 == 1:
    X_test = X_test[:-1,:]
    y_test = y_test[:-1]

training_data = X_train
X_train = get_data(X_train)
X_test_2D = X_test
X_test = get_data(X_test)

batch_size = 2
#batch_size = 3
input_dim = X_train.shape[-1] # 13
timesteps = X_train.shape[1] # 3


vae, enc, gen = create_lstm_vae(input_dim, timesteps=timesteps, batch_size=batch_size, intermediate_dim=32, 
                                latent_dim=2, epsilon_std=std_dev_feed)
history = vae.fit(X_train, X_train, epochs=10, batch_size=batch_size, shuffle=True,
                    validation_data=(X_test, X_test),verbose=1).history

latent_dim=2
#    z_grid = norm.ppf(emn)

predictions = vae.predict(X_test, batch_size=batch_size)
#dec_pred = gen.predict(z_grid.reshape(-1,latent_dim))
    # pick a column to plot.
print("[plotting...]")
print("x: %s, preds: %s" % (X_train.shape, predictions.shape))
plt.plot(X_test[:,0,1], label='test data')
plt.plot(predictions[:,0,1], label=' Vae predict')
#plt.plot(dec_pred[:,0], label=' Decoder predict')
plt.legend()
plt.show()

from UniVariate_TS_Output_3Dto2D import Convert_2D

pred_Ravel = Convert_2D(predictions)

'''
While using log cosh as error function for prediction
'''
mse = np.mean(np.power(X_test_2D - pred_Ravel, 2), axis=1)
#mse = pred_Ravel
#mse = mse.reshape(-1,)
plt.plot(mse)
plt.xlabel('Individual samples')
plt.ylabel('Reconstruction loss')
plt.show()

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test})
error_df.describe()

thres_array = []
roc_array = []
val_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1 ,1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 1.9, 2.0, 2.1, 2.2])
for i in val_range:
    threshold = error_df.reconstruction_error.mean()  + 0.4 * i * error_df.reconstruction_error.std()
    thres_array.append(threshold)
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    error_df['boolean_reconstruction'] = y_pred
    fpr, tpr, threshold_s = roc_curve(error_df.true_class, error_df.boolean_reconstruction)
    roc_auc = auc(fpr, tpr)
    print('For index == %f  threshold == % 0.5f  the ROC = % 0.2f'  %(i, threshold, roc_auc))
    roc_array.append(roc_auc)

indx_max = roc_array.index(max(roc_array))

threshold_max_roc = thres_array[indx_max]

threshold = threshold_max_roc

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
error_df['boolean_reconstruction'] = y_pred
fpr, tpr, threshold_s = roc_curve(error_df.true_class, error_df.boolean_reconstruction)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


LABELS = ["Normal", "Anamoly"]


#y_pred = [1 if (e > threshold_u or e < threshold_d) else 0 for e in error_df.reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

tp = conf_matrix[0,0]
tn = conf_matrix[1,1]
fp = conf_matrix[1,0]
fn = conf_matrix[0,1]

# precision tp / (tp + fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
print('Accuracy is%f' %accuracy)

precision = tp/(tp+fp)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = tp / (tp + fn)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 =  2 * (precision * recall) / (precision + recall)
print('F1 score: %f' % f1)

FN = error_df.loc[(error_df['true_class'] == 0) & (error_df['reconstruction_error'] == 1) ]
#print(FN)
FP = error_df.loc[(error_df['true_class'] == 1) & (error_df['reconstruction_error'] == 0) ]
#print(FP)
x = np.c_[fpr,tpr]
dataset = pd.DataFrame({'FPR':fpr,'TPR':tpr})
print('Threshold is: %f' % threshold)

print('ROC = {0:0.2f}'.format(roc_auc))


def fit_sub(df_n_train, df_n_test, mdl, cols_features):

    X_train = df_n_train[cols_features].values
    y_train = df_n_train['FAILURE_NEAR'].values > 0
    
    X_test = df_n_test[cols_features].values
    y_test = df_n_test['FAILURE_NEAR'].values > 0

    print(X_train.shape)
    if mdl is None:
        mdl = sk.pipeline.Pipeline([
            ('scaler', sk.preprocessing.MinMaxScaler()),
            ('regression', sk.svm.SVC(gamma='scale')),
        ])


    y_cv = sk.model_selection.cross_val_predict(
        mdl,
        X_train,
        y_train,
        cv=5
    )

    mdl.fit(X_train, y_train)
    y_test_p = mdl.predict(X_test) > 0
    
    fpr, tpr, threshold_s = roc_curve(y_test, y_test_p)
    roc_auc = auc(fpr, tpr)

    print("cv test accuracy:       %s" % sk.metrics.accuracy_score(y_cv, y_train))
    print("testing units accuracy: %s" % sk.metrics.accuracy_score(y_test_p, y_test))
    print("testing units ROC: %s" % roc_auc)
    res_full = {
        'dy_train_cv': np.logical_not(np.logical_and(y_cv, y_train)),
        'dy_test': np.logical_not(np.logical_and(y_test_p, y_test)),
        'y_train_p_cv': y_cv,
        'y_test_p': y_test_p,
    }
    
    return res_full