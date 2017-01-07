import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               use_spatial_batchnorm=False,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.num_filters = num_filters
    self.reg = reg
    self.use_spatial_batchnorm = use_spatial_batchnorm
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    (C, H, W) = input_dim
    
    stride_conv1 = 1
    pad = (filter_size - 1) / 2
    
    stride_pool1 = 2
    pool_height = 2
    pool_width = 2
    
    
    # initialize 1st layer, (conv_relu_pool)
    self.params['W'+str(1)] = np.random.randn(num_filters, C, filter_size, filter_size) * \
                              weight_scale
    self.params['b'+str(1)] = np.zeros(num_filters)
    H_out = (H + 2 * pad - filter_size) / stride_conv1 + 1
    W_out = (W + 2 * pad - filter_size) / stride_conv1 + 1
    H_out = H_out / 2
    W_out = W_out / 2
    
    if self.use_spatial_batchnorm:
      gamma = np.ones(num_filters)
      beta = np.zeros(num_filters)
      self.params['gamma'] = gamma
      self.params['beta'] = beta
      bn_param = {}
      if 'eps' not in bn_param:
        bn_param['eps'] = 1e-8
      if 'momentum' not in bn_param:
        bn_param['momentum'] = 0.9
      if 'running_mean' not in bn_param:
        bn_param['running_mean'] = np.zeros(self.num_filters)
      if 'running_var' not in bn_param:
        bn_param['running_var'] = np.zeros(self.num_filters)
      self.bn_param = bn_param
    
    if False:
      print '1st layer H_out: {}, W_out:{}'.format(H_out, W_out)
    
    # initialize 2nd layer (affine_relu)
    self.params['W'+str(2)] = np.random.randn(num_filters * H_out * W_out, hidden_dim) * \
                              weight_scale
    self.params['b'+str(2)] = np.zeros(hidden_dim)
    
    # initialize 3rd layer (affine)
    self.params['W'+str(3)] = np.random.randn(hidden_dim, num_classes) * \
                              weight_scale
    self.params['b'+str(3)] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    bn_param = self.bn_param
    if y is None:
      bn_param['mode'] = 'test'
    else:
      bn_param['mode'] = 'train'
      
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache = {}
    
    # 1st layer, conv_relu_pool
    if False:
      print 'X.shape'
      print X.shape
    if not self.use_spatial_batchnorm:
      out, cache[1] = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    else:
      gamma = self.params['gamma']
      beta = self.params['beta']
      out, cache[1] = conv_norm_relu_pool_forward(X, W1, b1, gamma, beta, conv_param, bn_param, pool_param)
    
    # 2nd layer, affine_relu
    if False:
      print 'in2.shape'
      print out.shape
      print 'W2.shape'
      print W2.shape
    out, cache[2] = affine_relu_forward(out, W2, b2)
    # 3nd layer, affine
    scores, cache[3] = affine_forward(out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, grad = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W3 * W3) + np.sum(W2 * W2) + np.sum(W1 * W1))
    
    # backward
    # layer 3 affine
    dout, dW3, db3 = affine_backward(grad, cache[3])
    grads['W3'] = dW3
    grads['b3'] = db3
    # layer 2 affine relu
    dout, dW2, db2 = affine_relu_backward(dout, cache[2])
    grads['W2'] = dW2
    grads['b2'] = db2
    # layer 1 conv relu pool
    if not self.use_spatial_batchnorm:
      dout, dW1, db1 = conv_relu_pool_backward(dout, cache[1])
      grads['W1'] = dW1
      grads['b1'] = db1
    else:
      dout, dW1, db1, dgamma, dbeta = conv_norm_relu_pool_backward(dout, cache[1])
      grads['W1'] = dW1
      grads['b1'] = db1
      grads['gamma'] = dgamma
      grads['beta'] = dbeta
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
