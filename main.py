# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:56:26 2018

@author: piscesmo
"""

import scipy.io as sio # 用来读取matlab数据
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Plot
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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

# Close-set
# LOSO per subject
sample_total = len(x_total)
counter = 1
y_labels = []
y_scores = []
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
        train_x, train_labels = x[train], labels[train]
        test_x, test_labels = x[test], labels[test]

        # Train random forest
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1)
        clf.fit(train_x, train_labels.ravel())

        # Evaluation
        predict_prob = clf.predict_proba(test_x)
        print('[{:0d}/{:0d}] Probi. of positive class: {} Real: {}'.format(
            counter, sample_total, predict_prob[0][1], test_labels[0][0]))
        counter += 1

        # Store the results
        y_labels += [test_labels[0][0]]
        y_scores += [predict_prob[0][1]]


y_labels = np.asarray(y_labels)
y_scores = np.asarray(y_scores)

fpr, tpr, thresholds = roc_curve(y_labels, y_scores, drop_intermediate=True)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
roc_auc = auc(fpr, tpr)

x = fpr
y = tpr
fig = plt.figure(figsize=(6, 5))
title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
axis_font = {'fontname': 'Arial', 'size': '14'}
plt.plot(x, y, label='EER={:.2f}%, AUC={:.2f}%'.format(eer*100, roc_auc*100), color='r', lw=2)
plt.title("ROC Curve", **title_font)
plt.xlim(0.0, 1.0)
plt.xlabel('False Positive Rate', **axis_font)
plt.ylim(0.0, 1.0)
plt.ylabel('True Positive Rate', **axis_font)
plt.grid(True)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()

plt.show()
filename = 'result.pdf'
fig.savefig(filename, format='pdf', dpi=1200)