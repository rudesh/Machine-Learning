
from builtins import range
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #pass
  
    #getting the dictionary keys
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')

    #input data shape
    N, C, H, W = x.shape
    
    #filter weights of shape
    F, C, HH, WW = w.shape

    #padding the tensor acording to the hint given
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))

    #computing the return tuple
    H_out = 1 + int((H + 2 * pad - HH) / stride)
    W_out = 1 + int((W + 2 * pad - WW) / stride)

    #broadcasting the out data according to the values we got above
    out = np.zeros([N, F, H_out, W_out])

    
    
    #finally looping convolution, each loop represents their own conditions
    
    #each image in the input x
    for img in range(N):
      
      #each filter in the input w
      for fil in range(F):
        
        #each output height
        for ht in range(H_out):
          
          #each output width
          for wd in range(W_out):
            
            #computing output data, shape (N, F, H_out, W_out)
            out[img, fil, ht, wd] = np.sum(padded_x[img, :, ht * stride:ht * stride + HH, wd * stride:wd * stride + WW] * w[fil, :]) + b[fil]
            
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    #pass
    
    #explore incoming cache and dout to variables
    x, w, b, conv_param = cache
    N, F, H_out, W_out = dout.shape
    
    #getting the dictionary keys
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')

    #input data shape
    N, C, H, W = x.shape
    
    #filter weights of shape
    F, C, HH, WW = w.shape

    #padding the tensor acording to the hint given
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))
    
    
    #broadcast gradient outputs to relavant shaspes with zeros
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    #also another extra padded dx
    padded_dx = np.zeros_like(padded_x)
    
    
    #computing gradients with respect to their tuples
    
    #each image in the input x
    for img in range(N):
      
      #each filter in the input w
      for fil in range(F):
        
        #computing db
        db[fil] += np.sum(dout[img, fil, :, :])
        
        #each output height
        for ht in range(H_out):
          
          #each output width
          for wd in range(W_out):
            
            #computing dx with using the pad
            padded_dx[img, :, ht * stride:ht * stride + HH, wd * stride:wd * stride + WW] += dout[img, fil, ht, wd] * w[fil, :, :, :]
            
            #removing padding
            dx = padded_dx[:, :, pad:H + pad, pad:W + pad]
            
            #computing dw
            dw[fil, :, :, :] += dout[img, fil, ht, wd] * padded_x[img, :, ht * stride:ht * stride + HH, wd * stride:wd * stride + WW]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    #pass
    
    #as mentioned in the description we can explore incoming x and pool_params as below
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')
    N, C, H, W = x.shape
    
    #computing the return tuple dimensions
    H_out = 1 + int((H - pool_height) / stride)
    W_out = 1 + int((W - pool_width) / stride)
    
    #broadcasting output as according to what we calculated above
    out = np.zeros([N, C, H_out, W_out])
    
    #each image in the input x
    for img in range(N):
      
      #each channel in the input x
      for ch in range(C):
        
        #each output row
        for row in range(H_out):
          
          #each output column
          for col in range(W_out):
            
            #computing output data shape (N, F, H_out, W_out)
            out[img, ch, row, col] = np.max(x[img, ch, row * stride:row * stride + pool_height, col * stride:col * stride + pool_width])
            
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    #pass
    
    #as mentioned in the description we can explore incoming cache and dout as below
    x, pool_param = cache

    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')
    
    N, C, H, W = x.shape
    a, b, H_dout, W_dout = dout.shape
    
    #broadcasting dx to be as same as input x
    dx = np.zeros_like(x)
    
    
    
    #each image in the input x
    for img in range(N):
      
      #each channel in the input x
      for ch in range(C):
        
        #each output row
        for row in range(H_dout):
          
          #each output column
          for col in range(W_dout):
            
             #argmax returns the linear index of the max of each
             maximum = np.argmax(x[img, ch, row * stride:row * stride + pool_height, col * stride:col * stride + pool_width])
             
             # Using unravel_index convert this linear index to matrix coordinate.
             max_coord = np.unravel_index(maximum, [pool_height, pool_width])
                    
             # Only backprop the dout to the max location.
             dx[img, ch, row * stride:row * stride + pool_height, col * stride:col * stride + pool_width][max_coord] = dout[img, ch, row, col]
            
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx