from numpy import zeros, array, exp, ones, dot, vectorize, concatenate, packbits, uint8, float_, append, arange
from numpy.random import rand, permutation
import matplotlib.pyplot as plt
from decimal import Decimal

class NeuralNet(object):
  """ define a neural network
  n_in -> number of input vectors
  n_h1, n_h2 -> number of neurons in hidden layer
  n_out -> number of neuron in output layer, how many classes?
  eta -> learning rate
  a -> mean for initializing weights and biases
  """
  def __init__(self, n_in, n_h1, n_h2, n_out, eta, a):
    # assign variables from the constructor
    self.n_in = n_in
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.n_out = n_out
    self.eta = Decimal(str(eta))
    self.a = Decimal(str(a))
    self.init()

  def init(self):
    # preallocate storage and initialize weights and biases
    # self.x_in = array(zeros((n_in, 1)), dtype=decimal.Decimal)
    self.w_h1 = array(-self.a + (2 * self.a) * self.dwrap(rand(self.n_h1, self.n_in).astype(Decimal)))
    self.b_h1 = array(-self.a + (2 * self.a) * self.dwrap(rand(self.n_h1, 1).astype(Decimal)))
    self.w_h2 = array(-self.a + (2 * self.a) * self.dwrap(rand(self.n_h2, self.n_h1).astype(Decimal)))
    self.b_h2 = array(-self.a + (2 * self.a) * self.dwrap(rand(self.n_h2, 1).astype(Decimal)))
    self.w_out = array(-self.a + (2 * self.a) * self.dwrap(rand(self.n_out, self.n_h2).astype(Decimal)))
    self.b_out = array(-self.a + (2 * self.a) * self.dwrap(rand(self.n_out, 1).astype(Decimal)))
    # self.d_out = array(zeros((n_out, 1)), dtype=decimal.Decimal)
    # self.dexp = vectorize(lambda x: x.exp(), otypes=[Decimal])
    # self.sig = vectorize(lambda x: 1 / (1 + exp(-x)), otypes=[Decimal],
        # doc='Vectorized logistic sigmoid function')
    self.assign = vectorize(lambda x, y: 1 if x >= y else 0, otypes=[uint8],
        doc='Returns a vector that returns true if an element is g/eq than 0.5, vectorized')

  def dwrap(self, arr):
    i, j = arr.shape[0], arr.shape[1]
    for x in range(i):
      for y in range(j):
        arr[x, y] = Decimal(str(arr[x, y]))
    return arr

  def train(self, X, Y, max_epoch, **kwargs):
    totalerr = self.dwrap(zeros((max_epoch, 1)).astype(Decimal))
    N = X.shape[0]
    for q in range(max_epoch):
      p = permutation(N) # shuffle patterns e.g. {7,3,5,1,2,4,6,0}
      for n in range(N):
        nn = p[n] # get random number from p
        # read data
        if(kwargs['random']):
          x_in = array([X[nn, :]]).astype(Decimal).T
          d_out = array([Y[nn, :]]).astype(Decimal).T
        else:
          x_in = array([X[n, :]]).astype(Decimal).T
          d_out = array([Y[n, :]]).astype(Decimal).T
        # forward prop
        # input to hidden layer 1
        v_h1 = dot(self.w_h1, x_in) + self.b_h1
        y_h1 = 1 / (1 + exp(-1 * v_h1))
        # print y_h1.shape
        # input to hidden layer 2
        v_h2 = dot(self.w_h2, y_h1) + self.b_h2
        y_h2 = 1 / (1 + exp(-1 * v_h2))
        # output layer
        # print y_h2.shape
        v_out = dot(self.w_out, y_h2) + self.b_out
        out = 1 / (1 + exp(-1 * v_out))
        # print out.shape

        # backward propagation
        # compute error
        err = d_out - out
        # compute gradient in output layer
        delta_out = err * out * (1-out)
        # compute gradient in hidden layer 2
        delta_h2 = y_h2 * (1-y_h2) * (self.w_out.T.dot(delta_out))
        # compute gradient in hidden layer 1
        delta_h1 = y_h1 * (1-y_h1) * (self.w_h2.T.dot(delta_h2))
        # update weights and biases in output layer
        self.w_out += self.eta * delta_out.dot(y_h2.T)
        self.b_out += self.eta * delta_out
        # update weights and biases in hidden layer 2
        self.w_h2 += self.eta * delta_h2.dot(y_h1.T)
        self.b_h2 += self.eta * delta_h2
        # update weights and biases in hidden layer 2
        self.w_h1 += self.eta * delta_h1.dot(x_in.T)
        self.b_h1 += self.eta * delta_h1
      # get totalerr after one epoch
      totalerr[q] += sum(err.T.dot(err))
      if (q+1) % 500 == 0:
        print 'totalerr at', q, '-',
        print totalerr[q]
      # if totalerr less than epsilon then end training
      if totalerr[q] < 0.001:
        break
    return (q, totalerr)

  def fprop(self, x_in):
    # input to hidden layer 1
    v_h1 = dot(self.w_h1, x_in) + self.b_h1
    y_h1 = 1 / (1 + exp(-1 * v_h1))
    # print y_h1.shape
    # input to hidden layer 2
    v_h2 = dot(self.w_h2, y_h1) + self.b_h2
    y_h2 = 1 / (1 + exp(-1 * v_h2))
    # output layer
    # print y_h2.shape
    v_out = dot(self.w_out, y_h2) + self.b_out
    out = 1 / (1 + exp(-1 * v_out))
    return out

  def test(self, X, Y):
    nn_output = zeros(Y.shape).astype(uint8)
    # print type(nn_output[0,0])
    for n in range(0, X.shape[0]):
      # read data
      x_in = array([X[n, :]]).astype(Decimal).T
      d_out = array([Y[n, :]]).astype(Decimal).T
      # forward prop
      out = array(self.fprop(x_in).T)
      output = self.assign(out, 0.5) # after comparing, determined which class is this
      nn_output[n, :] = output;
    return nn_output
    #   print x_in.T, packbits(append(array([0,0,0,0,0]), output))
    #   print array(x)
    #   print packbits(append(array([0,0,0,0,0]), output))

if __name__ == '__main__':
  max_epoch = 30000
  myann = NeuralNet(3, 7, 5, 3, 0.1, 0.1)
  X = array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
  Y = array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0]])
  (end, nn_error) = myann.train(X, Y, max_epoch, random=True)
  nn_output = myann.test(X, Y)
  print 'Total bits with error', sum(sum(abs(Y-nn_output)))
  print 'Ran through', end, 'epochs with final error of', nn_error[end][0]
  plt.plot(arange(0, max_epoch, 1), nn_error)
  plt.xlabel('epochs')
  plt.ylabel('error')
  plt.title('Neural network training error over time')
  plt.show()
