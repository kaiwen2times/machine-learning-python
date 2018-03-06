import numpy as np
import pdb

def AdaBoostError(weights, classifier, localXdata, localGT):
    # function [errorAmt, alpha] = AdaBoostError(weights, classifier,localXdata, localGT)
    #  Compute AdaBoost Error across a set of training samples
    # Input:
    #   weights- one weight per input sample, nx1
    #   classifier- a struct containing the fields feature, thresh, and polarity
    #   localXdata- the X values of input sample, nxD
    #   localGT- the ground truth for each input sample, nx1
    # Output:
    #   error_amt- current error
    #   alpha- AdaBoost alpha value
    #   predict- prediction class [-1,+1] of each input sample


    # make a predition using the classifier on our data
    # predict = (2 * (Xdata(:,feature) < T) - 1) * polarity
    predict = (2 * ( localXdata[:, classifier['feature']] < classifier['thresh'] ) - 1) \
              * classifier['polarity']

    # form a [0,1] vector mistakes, mistakes will have one entry for each
    mistakes = (predict != localGT).nonzero()

    # weight these mistakes
    weighted_mistakes = np.dot(mistakes, weights)

    # relative error
    error_amt = np.sum(weighted_mistakes) / np.sum(weights)

    # alpha as per AdaBoost
    alpha = 0.5 * np.log((1 - error_amt) / error_amt)

    return error_amt, alpha, predict
