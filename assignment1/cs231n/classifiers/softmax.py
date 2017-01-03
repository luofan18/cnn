import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  if False:
    for i in xrange(num_train):
      X_i =  X[i]
      score_i = X_i.dot(W)
      stability = -score_i.max()
      exp_score_i = np.exp(score_i+stability)
      exp_score_total_i = np.sum(exp_score_i , axis = 0)
      for j in xrange(num_class):
        if j == y[i]:
          dW[:,j] += -X_i.T + (exp_score_i[j] / exp_score_total_i) * X_i.T
        else:
          dW[:,j] += (exp_score_i[j] / exp_score_total_i) * X_i.T
      numerator = np.exp(score_i[y[i]]+stability)
      denom = np.sum(np.exp(score_i+stability),axis = 0)
      loss += -np.log(numerator / float(denom))
  
  else:
    for i in xrange(num_train):
      score = X[i].dot(W)
      
      # shift the score 
      logC = - np.max(score)
      score_shift = score + logC
      
      # take the exponential
      score_shift = np.exp(score_shift)

      prob = np.exp(score_shift) / np.sum(np.exp(score_shift))
      loss += - np.log(prob[y[i]])
      
      for j in xrange(num_class):
        if j == y[i]:
          dW[:,j] += - ((1 - prob[y[i]]) * X[i]).T
        else:
          dW[:,j] += (prob[j] * X[i]).T
        
  dW /= num_train
  dW += reg * W
  
  
  loss /= num_train
  
  loss += 0.5 * reg * np.sum(W * W)
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
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  score = X.dot(W)
  logC = -score.max(axis = 1)
  
  score_shift = score + logC.reshape((-1,1))
  score_shift = np.exp(score_shift)
  prob = score_shift / np.sum(score_shift, axis = 1).reshape((-1,1))
  
  loss = np.sum( - np.log( prob[range(num_train), y]))
  
  # print prob[range(num_train), y]
  # print prob[range(num_train), y].shape
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  prob[range(num_train), y] -= 1
  dW += prob.T.dot(X).T
  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

