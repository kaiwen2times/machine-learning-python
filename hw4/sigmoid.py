import numpy as np
import pdb

def sigmoid(z):
    # sigmoid Compute sigmoid functoon
    # g = sigmoid(z) computes the sigmoid of z.
    pdb.set_trace()
    g = 1 / (1 + np.exp(-z))

    return g