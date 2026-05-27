import sys
import numpy as np
from sklearn.svm import SVC as svm
from sklearn.neural_network import MLPClassifier as ann
import math
import matplotlib.pyplot as plt

#read test data for sklearn svm
train_data = np.loadtxt('train_skdata', delimiter=' ')
X = train_data[0:, 1:]
y = train_data[0:, 0]
test_data = np.loadtxt('test_skdata', delimiter=' ')
testX = test_data[0:, 1:]
testY = test_data[0:, 0]

# print 'training ANN...'
# clf = ann(hidden_layer_sizes=(1000, 1000, 1000),
    # activation='tanh', solver='sgd',
    # learning_rate='adaptive', max_iter=1000,
    # shuffle=True, tol=1e-4, learning_rate_init=0.1,
    # verbose=True, momentum=0.3, early_stopping=True,
    # validation_fraction=0.1)
# clf.fit(X, y)
# predY = clf.predict(testX)
# residual = testY - predY
# acc = 1 - np.count_nonzero(residual) / float(residual.size)
# print 'accuracy:', acc

print 'getting SVs'
clf = svm(kernel='poly', degree=5, C=10000)
clf.fit(X, y)
print clf.score(testX, testY)

exit(0)

poly_d = range(1, 11)
coes = range(0, 11)
gammas = ['auto']

print 'training linear SVM...'
clf = svm(kernel='linear')
clf.fit(X, y)
print 'accuracy:', clf.score(testX, testY)

for deg in poly_d:
    for gamma in gammas:
        for coe in coes:
            print 'training poly: deg =', deg, ' gamma =', gamma, 'coef0 =', coe, 'SVM...'
            clf = svm(kernel='poly', degree=deg, gamma=gamma, coef0=coe)
            clf.fit(X, y)
            print 'accuracy:', clf.score(testX, testY)

for gamma in gammas:
    print 'training rbf =', gamma, 'SVM...'
    clf = svm(kernel='rbf', gamma=gamma)
    clf.fit(X, y)
    print 'accuracy:', clf.score(testX, testY)

for gamma in gammas:
    for coe in coes:
        print 'training sigmoid: gamma =', gamma, ' coef0 =', coe, 'SVM...'
        clf = svm(kernel='sigmoid', gamma=gamma, coef0=coe)
        clf.fit(X, y)
        print 'accuracy:', clf.score(testX, testY)
