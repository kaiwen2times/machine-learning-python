import numpy as np
import pdb

def DecisionStump(weights, trainXdata, trainGT):
    # input:
    #   weights. n x 1 matrix of weights
    #   trainXdata. n x r matrix, where each of n rows is a data sample with r
    #   feature values
    #   trainGT. nx1 class values. I will force to -1 and 1 if they aren't already
    # output:
    #   classifier a structure indicating the selected feature, the threshold
    #   and the classes (for above and below the threshold

    classifier = {}

    mn = np.amin(trainGT)
    mx = np.amax(trainGT)
    # normalize between 0 and 1
    ccc = (trainGT - mn) / (mx -mn)
    n0 = np.sum((ccc == 0))
    n1 = np.sum((ccc == 1))

    # for every feature, find the best threshold
    nf = trainXdata.size[1]
    ns = trainXdata.size[0]

    for i in range(0, nf):
        # for each feature
        sorted = trainXdata[:, i].sort(axis=0)
        indices = trainXdata[:, i].argsort(axis=0)
        labs = ccc[indices]
        wts = weights[indices]
        # count the number of correct classifications
        cc = np.cumsum(labs * wts)
        zz = np.flipud( np.cumsum( np.flipud( (1-labs) * wts ) ) )
        bz = zz[0]
        zz = [zz[1:-1]; 0]


        # classify as 1 below the threshold!
        #numCorrect= cc + nz + (cc - nlist);#number correct as a function of threshold
        numCorrect= cc + zz;#number correct as a function of threshold

        # will need to change to accomadate the weights...
        # now pick the best... I want the classification that is the furthest
        # from 50#!!
        nc = abs(numCorrect-(cc(end)+ bz)/2);
        indicator = ([diff(tc); 1]>0); # shows where categories end
        nc = nc.*indicator;

       # [tc cc zz numCorrect nc]
        #labs
        #numCorrect
        #nc
        [mmm,iii] = max(nc); # pick out the most extreme
        flag(i) = sign(numCorrect(iii) - (cc(end)+ bz)/2);  # tells the classification direction
        #numCorrect(iii)
        if(iii<fs)
            thresh(i) = .5*(tc(iii)+ tc(iii+1));
        else
            thresh(i) = tc(iii) + .0001;
        end

        # check the endpoints to see if all should be classified one way...
        all1 = cc(end);
        all0 = zz(end);
        if(all1>mmm+.5)
            #classify everything as 1.
            thresh(i) = tc(end) + .0001;
            flag(i) = 1;
            mmm = all1-.5;
        elseif (all0>mmm+.5)
            thresh(i) = tc(1) +.0001;
            flag(i) = -1;
            mmm = all0-.5;
        end
       # mmm
        perf(i) = mmm+.5;
    end
    #[flag;
    #perf;
    #thresh]
    # now finding the best classifier is easy:
    [mmm,iii] = max(perf)
    # iii is the best classifier.
    classifier.feature  = iii
    classifier.thresh   = thresh(iii)
    classifier.polarity = flag(iii)
    classifier.performance =perf(iii)
    performance = perf(iii);

    return classifier