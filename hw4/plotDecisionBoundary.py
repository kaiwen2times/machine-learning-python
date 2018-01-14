import numpy as np
import matplotlib.pyplot as plt
from mapFeature import mapFeature
import pdb

def plotDecisionBoundary(theta, X, y, degree):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    #   positive examples and o for the negative examples. X is assumed to be 
    #   a either 
    #   1) Mx3 matrix, where the first column is an all-ones column for the 
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    # plot Data
    index0 = np.where(y==0)[0]
    index1 = np.where(y==1)[0]
    
    plt.plot(X[index0,0], X[index0,1], 'ro')
    plt.plot(X[index1,0], X[index1,1], 'g+')

    if X.shape[1] <= 3:
        # only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.amin(X[:,1])-2,  np.amax(X[:,1])+2])

        # calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        # plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
        
    else:
        # here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u),len(v)))
                     
        # evaluate z = theta*x over the grid
        for i in range(0, len(u)):
            for j in range(0, len(v)):
                z[i,j] = mapFeature(u[i], v[j], degree) * theta
            # end
        # end

        z = z.T
        plt.contour(u, v, z, [0, 0], linewidth=2)
