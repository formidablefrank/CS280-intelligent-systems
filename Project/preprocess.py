import os
import numpy as np
out = open('train_skdata', 'w')
label = 0

print 'constructing training data...'

# construct training data
for root, dirs, fil in os.walk('.'):
    if root != '.' and root != './.idea' and root != './.idea/scopes':
        print root
        for f in fil:
            if 'sunglasses' not in f:
                print f
                tempFile = list(open(os.path.join(root, f), 'r').read().split())[4:]
                print >> out, label,
                for i in range(1, len(tempFile) + 1):
                    print >> out, tempFile[i-1],
                print >> out
        label += 1


label = 0
out.close()
out = open('test_skdata', 'w')

print 'constructing testing data...'

# construct testing data
for root, dirs, fil in os.walk('.'):
    if root != '.' and root != './.idea' and root != './.idea/scopes':
        print root
        for f in fil:
            if 'sunglasses' in f:
                print f
                tempFile = list(open(os.path.join(root, f), 'r').read().split())[4:]
                print >> out, label,
                for i in range(1, len(tempFile) + 1):
                    print >> out, tempFile[i-1],
                print >> out
        label += 1
label = 0

out.close()
