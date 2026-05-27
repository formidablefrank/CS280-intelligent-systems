from numpy import zeros, array, exp, ones, dot, vectorize, concatenate, packbits, unpackbits, uint8, float_, append, arange, loadtxt, genfromtxt, vstack, count_nonzero, sum, savetxt
from numpy.random import rand, permutation

def shuffle_in_unison(a, b):
  assert len(a) == len(b)
  p = permutation(len(a))
  return a[p], b[p]

trainX = loadtxt('training_set.csv', delimiter=',', dtype=float_)
trainY = loadtxt('training_labels_new.csv', delimiter='\n', dtype=uint8)
valX = loadtxt('validation_set.csv', delimiter=',', dtype=float_)
valY = loadtxt('validation_labels_new.csv', delimiter='\n', dtype=uint8)

X = vstack((trainX, valX))
Y = concatenate((trainY, valY), axis=0)

givenX = X[:3468,:]
givenY = Y[:3468]
genX = X[3468:,:]
genY = Y[3468:]

# print genY

genX, genY = shuffle_in_unison(genX, genY)
genX, genY = shuffle_in_unison(genX, genY)

# print genY

X = vstack((givenX, genX))
Y = concatenate((givenY, genY), axis=0)

trainX = X[:6500,:]
trainY = Y[:6500]
valX = X[6500:,:]
valY = Y[6500:]

savetxt('training_set_final.csv', trainX, delimiter=',', fmt='%.8f')
savetxt('training_labels_final.csv', trainY, delimiter='\n', fmt='%d')
savetxt('validation_set_final.csv', valX, delimiter=',', fmt='%.8f')
savetxt('validation_labels_final.csv', valY, delimiter='\n', fmt='%d')
