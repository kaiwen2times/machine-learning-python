import numpy as np
import decisionStump from decisionStump
import adaBoostError from adaBoostError

def myAdaBoost(trainCV, trainGT, numFeatures, testCV, testGT)
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
    nlist = np,arange(1, ns+1).T


    # each sample gets a weight r
    weights = trainGT * 0 + (1 / ns)

    for i in range(1, nf+1):
        # each iteration creates a classifier
        h1 = decisionStump(weights, trainCV, trainGT)   # train base learner
        errorAmt, alpha = adaBoostError(weights, h1, trainCV, trainGT)  # determine alpha
        h1.alpha = alpha;
        [newWeights,zz] = AdaBoostUpdateWeights(weights,h1,TrainXdata,TrainGT);    %update the weights
        h1.z = zz;                                                  %theoretical error bound
        classifiers{i} = h1;

        %Document the performance
        trainPred = AdaBoostClassifier(classifiers,TrainXdata);
        trainErr(i)= sum(trainPred~=TrainGT)/n;
        testPred = AdaBoostClassifier(classifiers,TestXdata);
        testErr(i)= sum(testPred~=TestGT)/size(TestXdata,1);
        if(i==1) errBound(i)= zz;
        else errBound(i) = zz*errBound(i-1);
        end

        weights = newWeights;
    # end

    errors.train = trainErr;
    errors.test = testErr;
    errors.eb = errBound;

    pred.train = trainPred;
    pred.test = testPred;

    return classifiers, errors, pred
