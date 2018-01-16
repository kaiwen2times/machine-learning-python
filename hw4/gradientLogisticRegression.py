import numpy as np
from sigmoid import sigmoid

def gradientLogisticRegression(theta, X, y, lam):
    # gradientLogisticRegression Compute gradient for logistic regression with regularization
    #    [grad] = gradientLogisticRegression(theta, X, y, lambda) computes using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters. 

    # number of training examples
    n = y.size
    
    # hypothesis
    h = np.dot(X, theta)
    
    # calculate gradients
    grad = (1 / n) * (np.dot(X.T, (sigmoid(h).reshape((-1,1)) - y)) + (theta.reshape((-1,1)) * lam) / n)
    grad[0] = (1 / n) * np.sum((sigmoid(h) - y) * X[:,0])
    
    return grad.flatten()
