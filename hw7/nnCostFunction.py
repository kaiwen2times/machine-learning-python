import pdb
import numpy as np
from sigmoid import sigmoid

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                     num_labels, X, y, lam):
    # nn_cost_function Implements the neural network cost function for a two layer
    # neural network which performs classification
    #   J, grad = nn_cost_function(nn_params, hidden_layer_size, num_labels,
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.

    # Reshape nn_params back into the parameters theta1 and theta2
    # for our 2 hidden-layer neural network
    theta1 = nn_params['Theta1'] #25x401
    theta2 = nn_params['Theta2'] #10x26
    
    num_samples = X.shape[0]
    J = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    # add bias to the input data
    bias = np.ones((num_samples, 1))
    X1 = np.concatenate((bias, X), axis=1)
    
    # forward propagate
    layer1 = sigmoid(theta1.dot(X1.T))
    bias = np.ones((1, layer1.shape[1]))
    layer2 = np.concatenate((bias, layer1), axis=0)
    output = sigmoid(theta2.dot(layer2))

    # reshape y to nn format, one hot encoding
    ynn = np.zeros((num_samples, num_labels))
    for i in range(num_samples):
        ynn[i, y[i] -1] = 1 # column #10 represents digit 0
    #end
    ynn = ynn.T
    
    # cost function - first without regularization
    J = (-1 / num_samples) * np.sum(np.sum( ynn * np.log(output) + (1 - ynn) * np.log(1 - output) ))

    return J
#end
