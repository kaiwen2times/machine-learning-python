import numpy as np
import pdb

def AdaBoostClassifier(classifier, localXdata):
# function [pred] = AdaBoostClassifier(classifier, local_xdata)
    #  Compute AdaBoost Classification across a set of training samples
    # Input:
    #   classifier- a struct containing the fields feature, thresh, and
    #                  polarity, one entry per threshold
    #   local_xdata- the X values of input sample, nxD
    # Output:
    #   predict- prediction class [-1,+1] of each input sample
    #

    # for every feature, find the best threshold
    nf = local_xdata.shape[1]  			# number of features
    ns = local_xdata.shape[0]  			# number of samples
    nc = len(classifier)                # the number of classifiers.
    totalPred = np.zeros(ns,1)
    predict = 0

    # for each classifier, make a guess on the input data
    for i in range(0, nc):
        # apply the classifier
        # extract the current iterations classifier
        cc = classifier[i]


        # make a prediction for each feature, threshold value, and polarity
        # the equation here is identical to to AdaBoostError.m, but now we also
        # need to multiply by the alpha value in our classifier entry
        predict = (2 * ( localXdata[:, cc['feature']] < cc['thresh'] ) - 1) \
                  * cc['polarity'] * cc['alpha']

        # keep running sum of predict
        totalPred = totalPred + predict
    # end

    # now convert threshold to [-1 1]
    # convert totalPred positive values to +1 and negative totalPred values to -1
    predict = (totalPred > 0).nonzero() * 2 - 1
    return predict
