import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    indicator = (scores - correct_class_score + 1) > 0
    
    for j in xrange(num_classes):
      if j == y[i]:
        # for the weight to calculate score of correct class, the error is the
        # number of the score do not meet the delta
        dW[:,j] += -(np.sum(indicator)-1) * X[i].T
        continue

      # for the weight to calculate the score of other class, the error is whether
      # this class meet the delta
      dW[:,j] += indicator[j] * X[i].T
      
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW += reg * W
  # dW are calculate above


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  
  score = X.dot(W)
  
  correct_class_score = np.zeros((num_train,1))
  correct_class_score = score[range(len(y)),y]
  #print 'min(y) is {}'.format(np.min(y))
  
  loss_matrix = score - correct_class_score[:,np.newaxis] + 1
  loss_matrix[loss_matrix<0] = 0
  
  # remove the contribution of correct_class_score
  loss = np.sum(loss_matrix) - num_train
  
  loss /= num_train
  
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  y_correct_mask = np.zeros(score.shape)
  
  y_uncorrect_mask = np.array(loss_matrix > 0)
  #print 'y_uncorrect_mask.shape'
  #print y_uncorrect_mask.shape
  
  y_uncorrect_mask[range(len(y)),y] = np.zeros(len(y))
  
  y_correct_mask[range(len(y)),y] = y_uncorrect_mask.sum(axis=1)
  
  
  #dW += y_uncorrect_mask.T.dot(X).T
  #dW -= y_correct_mask.T.dot(X).T
  
  dW += (y_uncorrect_mask - y_correct_mask).T.dot(X).T
  
  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
