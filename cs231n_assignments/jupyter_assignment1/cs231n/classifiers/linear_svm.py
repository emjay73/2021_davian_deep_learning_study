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
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # n_train x n_class
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss        += margin                
                dW[:, y[i]] += -X[i].T # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                
                dW[:, j]    += X[i].T  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * np.sum(2*W) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # up!

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n_data = X.shape[0]
    s = np.dot(X, W) # score
    s_gt = np.expand_dims(s[np.arange(n_data), y], 1) # gt score    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!           
    l = np.maximum( s - s_gt  + 1 , 0)                              # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!           
    l[np.arange(n_data), y] = 0

    loss = np.sum( l ) / n_data

    loss += reg* np.sum(W * W)
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


    # loss[i] => sum over J {s[i, j] - s[i, y[i]] + 1}  NOTE. j is in J{s[i, j] > s[i, y[i]] - 1, j!= y[i]}. otherwise, loss as j is 0
    # loss[i] / dW[:, j] =  (ds[i, j]/dW[:, j]) * (dloss[i]/ds[i, j])
    # dloss[i] / s[i, j] => 1     when j!=y[i] 
    #                    => -#J   when j==y[i]
    # (ds[i, j]/dW[:, j]) = d(X[i, :]W[:, j])/dW[:, j] = X[i, :].T
    # therefore, dloss[i]/dW[:, j] =   1*X[i, :].T when j!=y[i]
    #                              = -#J*X[i, :].T when j==y[i]

    # let's expand j
    # dloss[i]/dW =   X[i, :].T @ [row vector, 1 when j!=y[i], -#J*X[i, :].T when j==y[i]] 

    # let's expand i to N
    # dloss[i]/dW =   X.T @ [[1 when j!=y[i], -#J*X[i, :].T when j==y[i]] ... n times]

     # Compute gradient
    l[l > 0] = 1 # incorrect class mask # (500, 10)
    valid_l_count = l.sum(axis=1) # # (500, ) # num(incorrect class) == num(class with loss)
    l[np.arange(n_data),y ] -= valid_l_count # (500, )
    dW = (X.T).dot(l) / n_data

    # Regularization gradient
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
