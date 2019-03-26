
from builtins import object
import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


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
        self.reg = reg
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
        #pass
        
        #exploring input_dim as mentioned in the description
        C, H, W = input_dim
        
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        
        self.params['W2'] = weight_scale * np.random.randn(num_filters * int(H/2) * int(W/2), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
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
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        #pass
        
        #for this i have used my own code from previous hw with some modifications
        maxpool_out, combined_cache = conv_relu_pool_forward(X, W1, b1,  conv_param, pool_param)

        #for the first set of W1 and b1, calling the affine_forward(maxpool_out, w, b)
        affine_one_out, affine_one_cache = affine_forward(maxpool_out, W2, b2)
        
        #so taking the layer one output and calling relu forward for that
        relu_two_out, relu_two_cache = relu_forward(affine_one_out)

        #now calling affine_forward(x, w, b) method with above relu output and W2 b2 parameters
        affine_two_out, affine_two_cache = affine_forward(relu_two_out, W3, b3)

        scores = affine_two_out
        
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
        #pass
        
        #computing the data loss by calling softmax_loss
        #basically this returns the loss with its gradient, here y is none for test
        data_loss, data_loss_gradient = softmax_loss(scores, y)

        #factor of 0.5 to simplify the expression for the gradient
        #according to L2 we have to get the squared value of all the weights
        regularisation_W1 = 0.5 * self.reg * np.sum(self.params['W1'] ** 2)
        regularisation_W2 = 0.5 * self.reg * np.sum(self.params['W2'] ** 2)
        regularisation_W3 = 0.5 * self.reg * np.sum(self.params['W3'] ** 2)
        
        #computing the loss
        loss = data_loss + regularisation_W1 + regularisation_W2 + regularisation_W3
        
        #computing gradients by calling affine_backward
        #for the affine forward cache and data loss gradient above
        three_dx, three_dw, three_db = affine_backward(data_loss_gradient, affine_two_cache)
        
        #so basically these are the weights and bias for below
        #using grads directory to save them
        grads['W3'] = three_dw + self.reg * W3
        grads['b3'] = three_db
        
        #now we can calculate W2 and b2 using the gradient we got above
        relu_two_dx = relu_backward(three_dx, relu_two_cache)
        two_dx, two_dw, two_db = affine_backward(relu_two_dx, affine_one_cache)
        
        #so these are the weights and biases accordingly
        grads['W2'] = two_dw + self.reg * W2
        grads['b2'] = two_db

        #similarly
        one_dx, one_dw, one_db = conv_relu_pool_backward(two_dx, combined_cache)

        #so these are the weights and biases accordingly
        grads['W1'] = one_dw + self.reg * W1
        grads['b1'] = one_db
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads