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


dfdata = loadmat('data/pendigits.mat')
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

mn = ['Column'+ str(i) for i in range(0,17)]
#X11 = pd.DataFrame(X11[:,0:10], columns=mn)
X11 = pd.DataFrame(X11[:,0:17], columns=mn)

X11.rename(columns={'Column16': 'Class'},inplace = True)
'''Changing data type from float to int in DatFrame'''
X11.Class = X11.Class.astype(int)

NewData = X11


anamoly_dataset = NewData.loc[NewData['Class'] == 1]
normal_dataset = NewData.loc[NewData['Class'] == 0]


anamoly_dataset = anamoly_dataset.sample(frac=1)
normal_dataset = normal_dataset.sample(frac=0.2)

ano_frac = anamoly_dataset.shape[0]/(anamoly_dataset.shape[0]+normal_dataset.shape[0])

newdf = pd.concat([normal_dataset, anamoly_dataset], axis=0)
newdf = shuffle(newdf)
newdf = newdf.sample(frac=1).reset_index(drop=True)

import keras
from keras import backend as K
from keras import objectives, losses

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.model_selection import train_test_split


from DeepVAELogCosh import create_lstm_vae
from PCA_Wavelet import Meig_Zero, Meig
from Gcd_Factorise import computeGCD


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

def get_data(data):
    # read data from file
    
    timesteps = 4    
    dataX = []
    for i in range(len(data) - timesteps + 1):
        x = data[i:(i+timesteps)]
        dataX.append(x)
    return np.array(dataX)

X_train, X_test = train_test_split(newdf, test_size=0.2, random_state=RANDOM_SEED)

X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pca1', 'pca2'])

std_dev_feed1 = Meig_Zero(principalDf['pca1'])
std_dev_feed2 = Meig_Zero(principalDf['pca2'])

std_dev_feed = 0.8 * std_dev_feed1 + 0.2 * std_dev_feed2

X_train = get_data(X_train)
X_test_2D = X_test
X_test = get_data(X_test)

batch_size = computeGCD(X_train.shape[0], X_test.shape[0])

input_dim = X_train.shape[-1] # 13
timesteps = X_train.shape[1] # 3


vae, enc, gen = create_lstm_vae(input_dim, timesteps=timesteps, batch_size=batch_size, intermediate_dim=32, 
                                latent_dim=2, epsilon_std=std_dev_feed)
history = vae.fit(X_train, X_train, epochs=3, batch_size=batch_size, shuffle=True,
                    validation_data=(X_test, X_test),verbose=1).history
    

latent_dim=2

predictions = vae.predict(X_test, batch_size=batch_size)

testdata = K.variable(X_test)
predscore = K.variable(predictions)
Keraspred = keras.losses.logcosh(testdata,predscore)
PredEval = K.eval(Keraspred)

from UniVariate_TS_Output_3Dto2D import Convert_2D

pred_Ravel = Convert_2D(PredEval)

mse = pred_Ravel

error_df = pd.DataFrame({'reconstruction_error': mse, 'boolean_reconstruction': mse, 'true_class': y_test})

error_df.describe()
AllPositive = error_df.loc[error_df['true_class'] == 1]

AllNegative = error_df.loc[error_df['true_class'] == 0]

anomaly_count = error_df[error_df['true_class'] == 1].count()
print("Count of total Anamoly %s" %anomaly_count)


import seaborn as sns

#threshold = error_df.reconstruction_error.mean() + 0.65 * error_df.reconstruction_error.std()

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

thres_array = []
roc_array = []
val_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1 ,1.2, 1.3])
for i in val_range:
    threshold = error_df.reconstruction_error.mean()  + 1 * i * error_df.reconstruction_error.std()
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
print('Anomaly percentage %f' % ano_frac)
print('ROC = {0:0.2f}'.format(roc_auc))

