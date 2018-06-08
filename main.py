# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:56:26 2018

@author: piscesmo
"""

import scipy.io as sio # 用来读取matlab数据
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np


def nFold(n):
    folds = []
    base = list(range(n))
    for i in range(n):
        test = [i]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


#matlab 文件名
matfn = './data/features_EYE_W2_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_EYE_W3_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_EYE_W23_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_MOUTH_W2_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_MOUTH_W3_LPQ_TOP_180406.mat'
#matfn = 'E:/supplement/features_MOUTH_W23_LPQ_TOP_180406.mat'
data = sio.loadmat(matfn)
#data2 = sio.loadmat(matfn2)

x_total = data.get('fea')
targets_total = data.get('labelValue')
labels_total = data.get('labelFeedback')
subject_labels = data.get('labelSubject')
subject_unique = np.unique(subject_labels)

# LOSO per subject
for subject in subject_unique:
    mask = subject_labels.ravel() == subject

    # select the features, targets, labels from a subject
    x = x_total[mask]

    targets = targets_total[mask]
    labels = labels_total[mask]
    num_sample = len(x)
    folds = nFold(n = num_sample)
    for train, test in folds:
        # Split the train set and test set
        train_x, train_targets, train_labels = x[train], targets[train], labels[train]
        test_x, test_targets, test_labels = x[test], targets[test], labels[test]

        # Train random forest
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1)
        clf.fit(train_x, train_targets.ravel())

        # Evaluation
        predict_prob = clf.predict_proba(test_x)
        print('Predict: {} Target: {}'.format(predict_prob, test_targets))
