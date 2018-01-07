import numpy as np
from computeCost import computeCost

def gradientDescentMultiReg(Xdata, y, theta, alpha, num_iters, lam):
    
    # GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(Xdata, y, theta, alpha, num_iters) updates theta by 
    #   taking num_iters gradient steps with learning rate alpha
    # Input:
    #   Xdata- input data, size nxD
    #   Y- target Y values for input data
    #   theta- initial theta values, size Dx1
    #   alpha- learning rate
    #   num_iters- number of iterations 
    #       Where n is the number of samples, and D is the dimension 
    #       of the sample plus 1 (the plus 1 accounts for the constant column)
    # Output:
    #   theta- the learned theta
    #   J_history- The least squares cost after each iteration

    # Initialize some useful values
    n = y.shape[0]
    J_history = np.zeros(num_iters)
    theta_temp = np.zeros(theta.shape[0])

    for iter in range(num_iters):
        h = np.dot(Xdata, theta).reshape((-1,1))
        for i in range(theta.shape[0]):  
            theta_temp[i] = theta[i] - (alpha / n) * np.sum(np.multiply(h - y, Xdata[:,i].reshape((-1,1)))) + \
            (alpha / n) * np.absolute(theta[i] * lam)

        theta = np.copy(theta_temp)

    return theta