import numpy as np
from sigmoid import sigmoid

def costFunctionLogisticRegression(theta, X, y, lam):
    # costFunctionLogisticRegression Compute cost and gradient for logistic regression with regularization
    #    [J] = costFunctionLogisticRegression(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters. 

    # number of training examples
    n = y.size
    
    # hypothesis
    h = np.dot(X, theta)

    # logistic Regression Cost Function
    J = (-1 / n) * (np.dot(y.T, np.log(sigmoid(h))) + np.dot((1-y).T, np.log(1-sigmoid(h)))) + (lam / (2 * n)) * np.sum(theta[1:-1]**2)
    
    return J
