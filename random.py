# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:56:26 2018

@author: piscesmo
"""

import scipy.io as sio # 用来读取matlab数据
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

#matlab 文件名
matfn = 'E:/supplement/features_EYE_W2_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_EYE_W3_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_EYE_W23_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_MOUTH_W2_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_MOUTH_W3_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_MOUTH_W23_LPQ_TOP_180406.mat'
data = sio.loadmat(matfn)
#data2 = sio.loadmat(matfn2)

x = data.get('fea')
y = data.get('labelValue')
y1 = y.ravel() #change column vector to an array

#split train set and validation set
train_x, test_x, train_y, test_y = cross_validation.train_test_split(x, y, random_state = 0)
# classification decision tree
clf = RandomForestClassifier(n_estimators = 100, criterion = 'gini')

#训练模型
s = clf.fit(train_x,train_y)

r = clf.score(test_x,test_y)
print (r)
prob_predict_test_y = clf.predict_proba(test_x)
