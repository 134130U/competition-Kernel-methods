path_to_utils = '/home/aims/kernel methods/Competition_kaggle/scripts'
import os
import sys
sys.path.append(path_to_utils)

import pandas as pd
import numpy as np
import models
from models import KernelLogisticRegression, KernelRidgeRegression,KernelSVM
#import KernelRidgeRegression
import Kernels
from Kernels import linear_kernel,quadratic_kernel, rbf_kernel
import kmer_featurization
import matplotlib.pyplot as plt
from numpy import linalg
import cvxopt
import cvxopt.solvers
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold



path = '/home/aims/kernel methods/Competition_kaggle/'
X = pd.read_csv(path+'data/Xtr.csv')
Y = pd.read_csv(path+'data/Ytr.csv')
test = pd.read_csv(path+'data/Xte.csv')

data = pd.concat([X['seq'],test['seq']], axis = 0)

y = Y['Bound'].to_numpy()
y = 2*y - 1 # transform from {0, 1} to {-1, 1}

k_mer = kmer_featurization.kmer_featurization(4)

kmer_feat = 10*k_mer.obtain_kmer_feature_for_a_list_of_sequences(data)

train = kmer_feat[:2000]
X_test = kmer_feat[2000:]

train_x, valid_x, train_y, valid_y = train_test_split(train,y,test_size=0.3)

kernel = 'quadratic'
lambd = 1e-5
sigma = .005
model = KernelRidgeRegression(
        kernel=kernel,
        lambd=lambd,
        sigma=sigma
    ).fit(train_x, train_y)

kf = KFold(n_splits=20)
scores = []
for train_index,test_index in kf.split(train):

    train_x, valid_x, train_y,valid_y = train[train_index],train[test_index], y[train_index], y[test_index]
    model.fit(train_x,train_y)
    ypred = model.predict(valid_x)
    print(np.mean(ypred == valid_y)*100)
    scores.append(np.mean(ypred == valid_y)*100)

print('final accuracy', np.mean(scores))

ypred = model.predict(valid_x)
correct = np.mean(ypred == valid_y)*100

# Prediction
y_pred = model.predict(X_test)
pred = (y_pred + 1)/2
pred = pred.astype(int)

test['Bound'] = pred
subm = test[['Id','Bound']]

subm.to_csv('Yte.csv',index= False)