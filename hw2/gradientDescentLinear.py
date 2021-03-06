import numpy as np
from computeCost import computeCost

def gradientDescentLinear(Xdata, y, theta, alpha, num_iters):
    
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

    for iter in range(num_iters):
        h = np.dot(Xdata, theta)
        theta0 = theta[0] - (alpha / n) * np.sum(h - y)
        theta1 = theta[1] - (alpha / n) * np.sum(np.multiply(h - y, Xdata[:,1].reshape((-1,1))))

        theta = np.array([theta0, theta1])

        # save the cost J in every iteration    
        J_history[iter] = computeCost(Xdata, y, theta)

    return (theta, J_history)