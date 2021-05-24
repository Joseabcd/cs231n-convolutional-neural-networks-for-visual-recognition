from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    data_dim = X.shape[1]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for m in range(num_classes):
          if m == y[i]:
            continue
          if (scores[m] - correct_class_score + 1 > 0):
            for l in range(data_dim):
              dW[l,m] += X[i,l]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                for l in range(data_dim):
                  dW[l,y[i]] -= X[i,l]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #***************************************************************************
    # A yet-even-more inefficient version of the implementation. It's not checked
    # because it turned out to be extremely slow.
    #for l in range(data_dim):
    #  for m in range(num_classes):        
    #    for i in range(num_train):
    #      correct_class_score = X[i].dot(W[:,y[i]])
    #      if y[i]==m:
    #        continue
    #      if X[i].dot(W[:,m])-correct_class_score+1>0:
    #        dW[l,m] += X[i,l]
    #      for j in range(num_classes):
    #        if j == y[i]:
    #            continue
    #        if y[i]!=m:
    #          continue
    #        if (X[i].dot(W[:,j])-correct_class_score+1>0):
    #          dW[l,m] -= X[i,l]
    # **************************************************************************
    
    dW /= num_train
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    margins = X.dot(W)
    correct_class_scores = np.reshape(margins[np.arange(num_train),y[np.arange(num_train)]],(-1,1)) 
    # (it's reshaped so as to do broadcasting along the columns; otherwise it'd below
    # be performed along the rows [recall broadcasting rule 1])
    margins -= correct_class_scores 
    margins += 1.0 # (we're in reality adding num_train ones in excess in there, but will subtract them immediately below)
    margins[np.arange(num_train),y[np.arange(num_train)]] -= 1.0
    loss = np.maximum(0.0,margins).sum() / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    indicator_margins_hat = (margins > 0.0).astype(int) # this is equivalent to applying the indicator function elementwise
    indicator_margins_hat = indicator_margins_hat[:,:,np.newaxis] # We want to broadcast this 2D array along a new 3rd dimension. If
    # we don't do this, numpy automatically will prepend ones (only one in this case) if we're to broadcast it with the rank-three array
    # in the line below (recall broadcasting rule 1), and thus it will automatically be reshaped inappropriately. (We want to prevent that
    # from happening and thus we manually put the shape 1 along the axis of our interest.)
    indicator_margins_hat =  indicator_margins_hat * np.rot90(X[:,:,None],1,(1,2))
    dW = dW + np.transpose(indicator_margins_hat.sum(axis=0)) # I'm summing here all the A_i_hat, as I defined them in my notes.
    np.add.at( dW, (np.arange(dW.shape[0]),y.reshape(-1,1)), -1.0 * indicator_margins_hat.sum(axis=1) ) # (the addend array is a 2D matrix
    # in which the i-th row contains the y[i]-th column of B_i_hat)

    dW /= num_train
    dW += 2.0 * reg * W

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
