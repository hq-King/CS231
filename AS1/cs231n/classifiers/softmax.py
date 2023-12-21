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

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores)
      correct_class_score = scores[y[i]]
      exp_sum = np.sum(np.exp(scores))
      loss += -correct_class_score + np.log(exp_sum) # np.log()以e为底
      for j in range(num_classes):
        if j == y[i]:
          dW[:, y[i]] += (np.exp(scores[y[i]])/exp_sum-1)*X[i]
        else:
          dW[:, j] += np.exp(scores[j])/exp_sum*X[i]
    loss /= num_train                      # 求平均损失
    loss += reg * np.sum(W * W)            # 损失加上正则化惩罚
    dW /= num_train                        # 求平均梯度
    dW += 2.0*reg*W
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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)                                                  # N*C 的矩阵
    scores -= np.max(scores, axis=1, keepdims=True)                    # 减去每行（每张图片对于每一类）的最大值
    correct_class_score = scores[range(num_train),y]
    exp_sum = np.sum(np.exp(scores), axis=1, keepdims=True)            # 按行求和，并保持为二维（列向量）
    loss = -np.sum(correct_class_score) + np.sum(np.log(exp_sum))      # 损失函数公式并求和
    loss = loss/num_train + reg * np.sum(W * W)
    
    med = np.exp(scores)/exp_sum         # 对于j!=yi的情况，dw=np.exp(scores[j])/exp_sum*X[i]
    med[range(num_train),y] -= 1         # 对于j=yi的情况，dw=(np.exp(scores[j])/exp_sum-1)*X[i]
    dW = X.T.dot(med)                    # 最后同时乘以 X[i]
    dW /= num_train
    dW += 2.0*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
