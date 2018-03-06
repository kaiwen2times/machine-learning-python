import numpy as np
import pdb
from AdaBoostError import AdaBoostError

def AdaBoostUpdateWeights(weights, classifier, localXdata, localGT):
    # function [newweights,zt] = AdaBoostUpdateWeights(inweights, classifier, localXdata, localGT)
    # Update weights for each iteration of AdaBoost
    # Input:
    #   inweights- one weight per input sample, nx1
    #   classifier- a struct containing the fields feature, thresh, and polarity
    #   localXdata- the X values of input sample, nxD
    #   localGT- the ground truth for each input sample, nx1
    # Output:
    #   updatedWeights- updated weights, one weight per input sample
    #   zt- sum of newweights

    # get the weighting of the classifier
    # call AdaBoostError to get alpha and predict
    err, alpha, predict = AdaBoostError(weights, classifier, localXdata, localGT)

    # calculate new weights
    updatedWeights = weights * np.exp(-alpha * predict * localGT)


    # normalize new weights
    zt = np.sum(updatedWeights)
    updatedWeights = updatedWeights / zt

    return updatedWeights, zt