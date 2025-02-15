import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
      scores = X[i].dot(W) 
      loss -= scores[y[i]] 
      loss += np.log(sum(np.exp(scores)))
      for j in xrange(num_classes):
          output = np.exp(scores[j])/sum(np.exp(scores)) 
          if j == y[i]:
              dW[:,j] += (-1 + output) *X[i] 
          else: 
              dW[:,j] += output *X[i] 

  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  dW = dW/num_train + reg* W 

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_score = scores[np.arange(num_train),y].reshape(num_train,1)
  exp_sum = np.sum(np.exp(scores),axis=1).reshape(num_train,1)
  loss += np.sum(np.log(exp_sum) - correct_class_score)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  margin = np.exp(scores) / exp_sum
  margin[np.arange(num_train),y] += -1
  dW = X.T.dot(margin)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


