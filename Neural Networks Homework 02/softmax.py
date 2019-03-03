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

    for i, x in enumerate(X):
        #############################################################################
        # TODO: Compute the softmax loss using explicit loops and store the result  #
        # in loss. If you are not careful here, it is easy to run into numeric      #
        # instability.                                                              #
        #############################################################################
        #pass
        
        #num_classes variable gets the weights W.shape np arrays 2nd index and num_train gets the 1st index
        num_classes = W.shape[1]
        num_train = X.shape[0]
        
        #initialised float variable loss to 0
        loss = 0.0
        
        #running 3 loops, 1 outer and 2 inner
        #for loop range runs all the values in the num_train from 0
        for a in range(num_train):
            
            #dot product of 1 x 10  inner product of vectors, weight
            score = X[a].dot(W)  
            correct_class_score = score[y[a]]
            exp_sum = 0
            
            
            for b in range(num_classes):
                
                #gets the exponential sum of score and correct class
                exp_sum += np.exp(score[b])
                softmax_activation = np.exp(correct_class_score) / exp_sum
                
                #calculating the log value
                loss += -1*np.log(softmax_activation)

            for c in range(num_classes):
                if c == y[a]:
                    dW[:, y[a]] += X[a, :] * (np.exp(score[y[a]])/exp_sum - 1)
                else:
                    dW[:, c] += X[a, :] * (np.exp(score[c])/exp_sum)

        # now the loss is a sum of all trainings, but it should to be an average, so getting the average
        loss /= num_train

        # regularization the loss.
        loss += reg * np.sum(W * W)
        dW = dW / num_train + 2 * reg * W
    
    
        #############################################################################
        # TODO: Compute the gradient using explicit loops and store it in dW.       #
        #############################################################################
        pass
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

    loss /= X.shape[0]
    dW /= X.shape[0]

    # Add regularization to the loss and gradients.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    #initialize the loss and gradient to zero
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability.                         #
    #############################################################################
    #pass
    score = np.dot(X,W)

    #calculate the exponential scores and normalize it beforehand with max as zero to avoid numerical instability
    #also keepdims true retains reduced dimensions
    exp_scores = np.exp(score - np.max(score, axis=1, keepdims=True))
    
    
    #softmax activation
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    
    #log loss of the correct class of each of our samples
    correct_logprobabilities = -np.log(probabilities[np.arange(X.shape[0]), y])

    #calculate the average loss
    loss = np.sum(correct_logprobabilities)/X.shape[0]

    #add regularization using the L2 norm which is a hyperparameter and controls the strength of regularization
    reg_loss = reg*np.sum(W*W)
    loss += reg_loss


    #for gradient dW, N x C
    dscores = probabilities
    dscores[np.arange(X.shape[0]), y] -= 1
    dscores /= X.shape[0]

    #dscores = X * W
    dW = X.T.dot(dscores)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    # Add regularization to the loss and gradients.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    return loss, dW
