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

print 'training ANN...'
clf = ann(hidden_layer_sizes=(5000,),
    activation='tanh', learning_rate='adaptive', max_iter=1000,
    shuffle=True, tol=1e-4, verbose=True)
clf.fit(X, y)
predY = clf.predict(testX)
print clf.score(textX, textY)
