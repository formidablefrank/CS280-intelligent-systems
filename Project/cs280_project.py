import os
import numpy as np
from sklearn.svm import SVC as svm
from sklearn.neural_network import MLPClassifier as ann
from sklearn.model_selection import GridSearchCV
import math
import matplotlib.pyplot as plt

#read test data for sklearn svm
train_data = np.loadtxt('train_skdata', delimiter=' ')
X = train_data[0:, 1:]
y = train_data[0:, 0]
test_data = np.loadtxt('test_skdata', delimiter=' ')
testX = test_data[0:, 1:]
testY = test_data[0:, 0]

C = [1, 10, 100, 1000]
poly_d = range(1, 11)
coes = range(0, 11)
gammas = np.linspace(0.1, 1, 10)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto'] + np.ndarray.tolist(gammas), 'C': C},
                    {'kernel': ['linear'], 'C': C},
                    {'kernel': ['poly'], 'C': C, 'degree': poly_d, 'gamma': ['auto'] + np.ndarray.tolist(gammas), 'coef0': coes},
                    {'kernel': ['sigmoid'], 'C': C, 'gamma': ['auto'] + np.ndarray.tolist(gammas), 'coef0': coes}]

scores = ['precision', 'recall']

for score in scores:
    print 'Tuning hyper-parameters for', score
    print
    clf = GridSearchCV(svm(), tuned_parameters, cv=4, n_jobs=-1, verbose=10)
    clf.fit(X, y)
    print 'Best parameters are found on development set:'
    print clf.best_params_
    print 'Grid scores on development set:'
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print 'Detailed classfication report:'
    print 'The model is trained on the full development set.'
    print 'The scores are computed on the full evaluation set.'
    predY = clf.predict(testX)
    print classfication_report(testY, predY)
    print
