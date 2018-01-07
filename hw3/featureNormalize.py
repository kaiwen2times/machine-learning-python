import numpy as np

def featureNormalize(X):
    # FEATURENORMALIZE Normalizes the features in X 
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    # You need to set these values correctly
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    for i in range(X.shape[0]):
        X_norm[i,:] = np.divide((X[i,:] - mu), sigma)

    return (X_norm, mu, sigma)