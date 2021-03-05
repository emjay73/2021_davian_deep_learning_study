from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n_data = X.shape[0]
    n_class = W.shape[1]

    # p = e^f[c] / sum_over_j_in_C(e^f[j])
    # softmax loss = -log ( e^f[c] / sum_over_j_in_C(e^f[j]) ) = -f[c] + log(sum_over_j_in_C(e^f[j]))
    # f[c] = X[i, :]@W[:, j]
    # dloss/dW[:, j] = (df[j]/dW[:, j])(dloss/df[j])
    # dloss / df[j] = when j==c : -1 + e^f[c]/sum_over_j_in_C(e^f[j])
    #               = when j!=c : e^f[c]/sum_over_j_in_C(e^f[j])
    # df[j]/dW[:, j] = X[i, :].T
    # therefore
    # dloss / dW[:, j] = when j==c : (-1 + p) * X[i, :].T
    #                  = when j!=c : (p) * X[i, :].T

    for i in range(n_data):
        logp = X[i]@W
        # logp = logp - np.max(logp)  # avoid numerical instability???
        p = np.exp(logp)/np.sum(np.exp(logp))

        # loss
        loss += -np.log(p[y[i]])

        # dw
        for j in range(n_class):
            dW[:, j] += X[i, :].T * p[j]
            # if j == y[i]:
            #    dW[:, j] -= X[i, :].T
        dW[:, y[i]] -= X[i, :].T
    loss /= n_data
    loss += reg*np.sum(W*W)

    dW /= n_data  # !!!!!!
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n_data = X.shape[0]

    # p = e^f[c] / sum_over_j_in_C(e^f[j])
    # softmax loss = -log ( e^f[c] / sum_over_j_in_C(e^f[j]) ) = -f[c] + log(sum_over_j_in_C(e^f[j]))
    # f[c] = X[i, :]@W[:, j]
    # dloss/dW[:, j] = (df[j]/dW[:, j])(dloss/df[j])
    # dloss / df[j] = when j==c : -1 + e^f[c]/sum_over_j_in_C(e^f[j])
    #               = when j!=c : e^f[c]/sum_over_j_in_C(e^f[j])
    # df[j]/dW[:, j] = X[i, :].T
    # therefore
    # dloss / dW[:, j] = when j==c : (-1 + p) * X[i, :].T
    #                  = when j!=c : (p) * X[i, :].T

    # let's expand j
    # dloss / dW = X[i, :].T @ [(p) ... (-1 + p)... (p)]
    #                                       ^
    #                                       |
    #                                     j==c

    # let's expand i
    # dloss / dW = X.T @ [[(p) ... (-1 + p)... (p)], ... n times]

    logp = X@W
    p = np.exp(logp)/np.expand_dims(np.sum(np.exp(logp), 1), 1)
    loss = np.sum(-np.log(p[np.arange(n_data), y]))
    loss /= n_data
    loss += reg*np.sum(W*W)

    pmat = p
    pmat[np.arange(n_data), y] -= 1
    dW = X.T @ pmat
    dW /= n_data  # !!!!!!
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
