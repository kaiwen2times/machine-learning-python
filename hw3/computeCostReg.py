import numpy as np

def computeCostReg(xData, y, theta, lam):
    # ComputeCostReg Compute cost for regularized linear regression
    #   J = computeCostReg(xData, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in xData and y
    # Input:
    #   Xdata, size nxD
    #   Theta, size Dx1
    #   Y, size nx1 
    #   lam is the regularization coefficient
    #       Where n is the number of samples, and D is the dimension 
    #       of the sample plus 1 (the plus 1 accounts for the constant column)
    # Output- J, the least squares cost
    
    h = np.dot(xData, theta)
    #Cost w/out regularization
    error_sum = np.sum(np.square(h - y))
    J = error_sum / (2 * y.shape[0])
    
    # Cost w/regularization
    J = J + (lam / (2 * y.shape[0])) * np.sum(np.square(theta[1:-1,:]))
    return J