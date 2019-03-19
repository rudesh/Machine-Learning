from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        # See also: http://cs231n.github.io/neural-networks-2/#init                #
        ############################################################################
        #pass
        #according to the link given in the question
        
        
        #computing W1 and W2
        
        #input_dim to hiidden_dim
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        
        #hidden_dim to num_classes
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        
        
        
        #computing b1 and b2
        
        #bias b1 for hidden_dim
        self.params['b1'] = np.zeros([hidden_dim])
        
        #bias b2 for num_classes
        self.params['b2'] = np.zeros([num_classes])
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #pass
        
        #for the first set of W1 and b1, calling the affine_forward(x, w, b)
        layer_one_out, layer_one_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        
        #so taking the layer one output and calling relu forward for that
        relu_one_out, relu_one_cache = relu_forward(layer_one_out)
        
        #now calling affine_forward(x, w, b) method with above relu output and W2 b2 parameters
        layer_two_out, layer_two_cache = affine_forward(relu_one_out, self.params['W2'], self.params['b2'])
        
        #running relu again with second layer out
        relu_two_out, relu_two_cache = relu_forward(layer_two_out)
        
        #finally getting scores
        scores = relu_two_out
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass
        
        #computing the data loss by calling softmax_loss
        #basically this returns the loss with its gradient, here y is none for test
        data_loss, data_loss_gradient = softmax_loss(scores, y)
        
        #factor of 0.5 to simplify the expression for the gradient
        #according to L2 we have to get the squared value of all the weights
        regularisation_W1 = 0.5 * self.reg * np.sum(self.params['W1'] ** 2)
        regularisation_W2 = 0.5 * self.reg * np.sum(self.params['W2'] ** 2)
        
        #computing the loss
        loss = data_loss + regularisation_W1 + regularisation_W2
        
        #computing gradients by calling affine_backward
        #for the relu layer two cache and data loss gradient above
        two_dx, two_dw, two_db = affine_backward(data_loss_gradient, layer_two_cache)
        
        #so basically these are the weights and bias for below
        #using grads directory to save them
        grads['W2'] = two_dw + self.reg * self.params['W2']
        grads['b2'] = two_db
        
        #now we can calculate W1 and b1 using the gradient we got above
        relu_dx = relu_backward(two_dx, relu_one_cache)
        one_dx, one_dw, one_db = affine_backward(relu_dx, layer_one_cache)
        
        #so these are the weights and biases accordingly
        grads['W1'] = one_dw + self.reg * self.params['W1']
        grads['b1'] = one_db
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        #pass
        
        
        #ignored batch normalisation as stated in piazza
        
        
        #initialising weights and biases for all fully connected layers
        for layer in range(int(self.num_layers)-1):
            self.params['W' + str(layer+1)] = weight_scale * np.random.randn(input_dim, hidden_dims[layer])
            self.params['b' + str(layer+1)] = np.zeros([hidden_dims[layer]])
            
            #setting the hidden dim of the current layer as the input of the next layer
            input_dim = hidden_dims[layer]
        
        #here setting the weight and the bias for the final layer 
        self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(input_dim, num_classes)
        self.params['b' + str(self.num_layers)] = np.zeros([num_classes])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #pass
    
    
        #ignored batch normalisation as stated in piazza
        
        #created three different lists to store cache outputs of fc layer
        affine_cache = {}
        dropout_cache = {}
        relu_cache = {}

        #layers count
        layers = self.num_layers
        
        #loop for every layer in the network
        for layer in range(layers-1):

            #sending the input X with weight and bias and gets out and cache tuple
            affine_out, affine_cache[str(layer+1)] = affine_forward(X, self.params['W'+str(layer+1)], self.params['b'+str(layer+1)])
            
            #passing the affine out to relu and compute
            tuple_out, relu_cache[str(layer+1)] = relu_forward(affine_out)
            
            #for dropout
            if self.use_dropout:
                tuple_out, dropout_cache[str(layer+1)] = dropout_forward(tuple_out, self.dropout_param)

            #setting the new input is either relu forward out or dropout forward out of a one loop run
            X = tuple_out

        #computing final score and cache
        scores, fc_final_cache = affine_forward(X, self.params['W'+str(layers)], self.params['b'+str(layers)])

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass
        
        #computing the data loss by calling softmax_loss
        #basically this returns the loss with its gradient, here y is none for test
        loss, data_loss_gradient = softmax_loss(scores, y)
        
        #computing gradients by calling affine_backward
        #for the relu layer two cache and data loss gradient above
        final_dx, final_dw, final_db = affine_backward(data_loss_gradient, fc_final_cache)
        
        #so basically these are the weights and bias for below
        #using grads directory to save them
        grads['W'+str(layers)] = final_dw + self.reg*self.params['W'+str(layers)]
        grads['b'+str(layers)] = final_db
        
        
        

        #going through each relu and layer to calculate gradients
        for layer in range(layers-1, 0, -1):
            
            #use dropout
            if self.use_dropout:
                final_dx = dropout_backward(final_dx, dropout_cache[str(layer)])

                
            gradient_relu = relu_backward(final_dx, relu_cache[str(layer)])
            
            final_dx, final_dw, final_db = affine_backward(gradient_relu, affine_cache[str(layer)])

            
            #storing gradients
            grads['W' + str(layer)] = final_dw + self.reg * self.params['W' + str(layer)]
            grads['b' + str(layer)] = final_db

            
            #computing loss
            #factor of 0.5 to simplify the expression for the gradient
            #according to L2 we have to get the squared value of all the weights
            loss = loss + 0.5 * self.reg * np.sum(self.params['W'+ str(layer)] ** 2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
