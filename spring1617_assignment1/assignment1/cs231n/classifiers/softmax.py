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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    scores = np.dot(X[i], W)
    shift_scores = scores - np.max(scores)
    p = np.exp(shift_scores[y[i]]) / np.sum(np.exp(shift_scores))
    loss_i = - np.log(p)
    loss += loss_i
    for j in range(W.shape[1]):
        p = np.exp(shift_scores[j]) / np.sum(np.exp(shift_scores))
        if j == y[i]:
            dW[ : , j] += (p - 1.0) * X[i].T
        else:
            dW[ : , j] += p * X[i].T

  loss /= X.shape[0]
  dW /= X.shape[0]
  loss += reg * np.sum(np.square(W))
  dW += reg * 2 * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  shift_scores = scores - np.array([np.max(scores, axis=1)]).T
  p = np.exp(shift_scores[range(X.shape[0]), y]) / np.sum(np.exp(shift_scores), axis=1)
  loss = np.sum(- np.log(p))
  loss /= X.shape[0]
  loss += reg * np.sum(np.square(W))
  
  p = np.exp(shift_scores) / np.array([np.sum(np.exp(shift_scores), axis=1)]).T
  p[range(X.shape[0]), y] -= 1
  dW = np.dot(X.T, p)
  dW /= X.shape[0]
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

