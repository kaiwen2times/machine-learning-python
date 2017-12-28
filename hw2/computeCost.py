import numpy as np

def computeCost(xdata, y, theta):
    h = np.dot(xdata, theta)
    error_sum = np.sum((h - y)**2)
    cost = error_sum / (2 * y.shape[0])
    return cost