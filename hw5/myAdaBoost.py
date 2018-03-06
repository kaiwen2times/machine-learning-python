import numpy as np
import pdb
from DecisionStump import DecisionStump
from AdaBoostError import AdaBoostError
from AdaBoostUpdateWeights import AdaBoostUpdateWeights
from AdaBoostClassifier import AdaBoostClassifier

def MyAdaBoost(trainXdata, trainGT, testXdata, testGT):
    # Function [classifiers, errors,pred] = myAdaBoost(TrainXdata,TrainGT,adaboost_numFeatures,TestXdata,TestGT)
    #  Implements the AdaBoost algorithm for decsion stump trees
    # Input:
    #   trainXdata- the X values of input training samples, nxD
    #   trainGT- the ground truth for each input training samples, nx1
    #   testXdata- the X values of input testing samples, nxD
    #   testGT- the ground truth for each input testing samples, nx1
    # Output:
    #   classifiers- vector of classifiers, each being a struct containing the
    #                    fields alpha, feature, performance, polarity,and
    #                    and thresh
    #   errors- AdaBoost classification errors for test and train set
    #   pred- AdaBoost classification result for test and train set

    # variables and params
    classifiers = []    # list of classifiers
    trainErr = []
    errBound = []
    errors = {}
    pred = {}

    nf = trainXdata.size[1]  # number of features, nf
    ns = trainXdata.size[0]   # number of samples, ns


    # each sample gets a weight
    weights = 1 / ns  # equal weights to start

    for i in range(0, nf):
        # each iteration creates a classifier.
        h1 = DecisionStump(weights, trainXdata, trainGT)
        errorAmt, alpha = AdaBoostError(weights, h1, trainXdata, trainGT)
        h1['alpha'] = alpha
        newWeights, zz = AdaBoostUpdateWeights(weights, h1, trainXdata, trainGT)
        h1['z'] = zz
        classifiers[i] = h1

        # document the performance
        trainPred = AdaBoostClassifier(classifiers, trainXdata)
        trainErr[i]= np,sum( (trainPred != trainGT).all() ) / ns
        testPred = AdaBoostClassifier(classifiers, testXdata)
        testErr[i]= np.sum( (testPred != testGT).all() ) / ns
        if i == 0:
            errBound[i]= zz
        else:
            errBound[i] = zz * errBound[i-1]
        # end
        weights = newWeights
    # end

    errors['train'] = trainErr
    errors['test'] = testErr
    errors['errorBound'] = errBound

    pred['train'] = trainPred
    pred['test'] = testPred

    return classifiers, errors, pred
