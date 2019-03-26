from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). For example,
    batch of 500 RGB CIFAR-10 images would have shape (500, 32, 32, 3). We 
    will reshape each input into a vector of dimension D = d_1 * ... * d_k,
    and then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    #pass
    
    #getting the quantity of the batch, index 0 represents the batch size of x
    number_of_images = x.shape[0]
    
    #reshape every image in the batch, literally flattenning the image
    reshaped_vector = x.reshape(number_of_images, -1)
    
    #transform reshaped_vector to output vector, adding biases
    #now the out vector is in shape(D, M)
    out = reshaped_vector.dot(w) + b
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass. Do not forget to reshape your #
    # dx to match the dimensions of x.                                        #
    ###########################################################################
    #pass
    
    #getting the quantity of the batch, index 0 represents the batch size of x
    number_of_images = x.shape[0]
    
    #reshape every image in the batch
    reshaped_vector = x.reshape(number_of_images, -1)
    
    #computing dx
    dx = dout.dot(w.T)
    
    #we have to rehsape the gradient x back to x's shape
    dx = dx.reshape(*x.shape)
    
    #computing dw
    dw = reshaped_vector.T.dot(dout)
    
    #finally computing db using numpy sum
    db = np.sum(dout, axis=0)
    
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    assert dx.shape == x.shape, "dx.shape != x.shape: " + str(dx.shape) + " != " + str(x.shape)
    assert dw.shape == w.shape, "dw.shape != w.shape: " + str(dw.shape) + " != " + str(w.shape)
    assert db.shape == b.shape, "db.shape != b.shape: " + str(db.shape) + " != " + str(b.shape)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    #pass
    
    #computing relu
    x_relu = x * (x >= 0)
    
    #getting the output
    out = x_relu
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    #pass
    
    #getting the inverse of x_relu, which used for forward
    inverse_x_relu = 1 * (x >= 0)
    
    #getting non negative elements
    dx = inverse_x_relu * dout
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        # HINT: http://cs231n.github.io/neural-networks-2/#reg                #
        #######################################################################
        #pass
        
        #getting inverse probability, for dropout purposes
        f_probability = 1.0 - p
        
        #getting the width and height of x
        size = x.shape
        
        #broadcasting a new array named mask using numpy random and uses width and height of x
        #also where neurons probability equal or higher than the probability p
        #as you the hint given by you
        mask = (np.random.rand(*size) < p)
        
        #getting the inverted dropout
        mask = mask / f_probability
        
        #drop
        out = mask * x

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        #pass
        
        #since the mode is test, just returning the input as it is
        out = x
        
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    p, mode = dropout_param['p'], dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        #pass
        
        dx = dout * mask
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
