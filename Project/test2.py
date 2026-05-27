from svmutil import *

y, x = svm_read_problem('train_data')
print 'training instances...'
model = svm_train(y, x, '-q -t 0')
#print 'saving model...'
#svm_save_model('model', model)
#print 'loading model...'
#model = svm_load_model('model')
a, b = svm_read_problem('test_data')
print 'testing data...'
classification, acc, val = svm_predict(a, b, model)
print classification
print acc
#print val
