import numpy as np
from DecisionStump import DecisionStump
from AdaBoostError import AdaBoostError
from AdaBoostUpdateWeights import AdaBoostUpdateWeights
from AdaBoostClassifier import AdaBoostClassifier

def AdaBoost(trainCV, trainGT, numFeatures, testCV, testGT):
    #  Implements the AdaBoost algorithm for decsion stump trees
    # Input:
    #   trainCV - the X values of input training samples, nxD
    #   trainGT - the ground truth for each input training samples, nx1
    #   numFeatures - number of thresholds to solve for
    #   testCV - the X values of input testing samples, nxD
    #   testGT - the ground truth for each input testing samples, nx1
    # Output:
    #   classifiers- vector of classifiers, each being a struct containing the
    #                    fields alpha, feature, performance, polarity,and
    #                    and thresh
    #   errors- AdaBoost classification errors for test and train set
    #   pred- AdaBoost classification result for test and train set
    #

    # for every feature, find the best threshold
    nf = trainCV.shape[1]  # number of features, nf
    ns = trainCV.shape[0]  # number of samples, ns
    claasifiers = {}


    # each sample gets a weight r
    weights = trainGT * 0 + (1 / ns)

    for i in range(0, nf):
        # each iteration creates a classifier
        h1 = decisionStump(weights, trainCV, trainGT)   # train base learner
        errorAmt, alpha = adaBoostError(weights, h1, trainCV, trainGT)  # determine alpha
        h1['alpha'] = alpha
        newWeights, zz = AdaBoostUpdateWeights(weights, h1, trainCV, trainGT)
        h1['z'] = zz
        classifiers['h1'] = h1

        # document the performance
        trainPred = AdaBoostClassifier(classifiers, trainCV)
        trainErr[i]= np.sum(trainPred~=trainGT) / n
        testPred = AdaBoostClassifier(classifiers, testCV)
        testErr[i]= np.sum(testPred~=TestGT) / testCV.shape[0]
        if(i == 0):
            errBound[i]= zz
        else:
            errBound[i] = zz * errBound[i-1]
        # end

        weights = newWeights
    # end

    errors['train'] = trainErr
    errors['test'] = testErr
    errors['eb'] = errBound

    pred['train'] = trainPred
    pred['test'] = testPred

    return classifiers, errors, pred
