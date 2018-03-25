function [confusionMatrix,accuracy,modelData] =  classify677_hwk8(X,y,options)
%function [confusionMatrix,accuracy,modelData] =  classify677_hwk8(Xdata,y,options)
% Input:
%      Xdata: A vector of feature, nxD, one set of attributes for each
%                sample (n samples x D dimensional data)
%      y: n x 1 ground truth labeling
%      options.method = {'LogisticRegression', 'KNN', 'SVM','ClassificationTree',
%                                 'BaggedTree', 'Boosting','Adaboost','nnet'}
%      options.numberOfFolds- the number of folds for k-fold cross validation
%      options.lambda - regularization lamda used in LogisticRegression
%      options.knn_k - number of nearest neighbors in KNN
%      options.adaboost_numFeatures- number of decision splits used in boosting
% Output:
%      confusionMatrix- an overall confusion matrix
%      accuracy- an overall accuracy value
%      modelData- parameters from model
%
%  CMPE-677, Machine Intelligence
%  R. Ptucha, 2016
%  Rochester Institute of Technology


[n,D] = size(X);
if length(y) ~= n
    error('X is nxD, and y is nx1');
end
numClasses = length(unique(y));
C = numClasses;

if (~exist('options','var'))
   options = [];
end

if  isfield(options,'method')
    method=options.method;
else
    method='KNN';
end
    
if strcmp(options.method,'LogisticRegression')
    Xdata = [ones(n, 1) X];
else
    Xdata = X;
end

if  isfield(options,'numberOfFolds')
   numberOfFolds=options.numberOfFolds;
else
    numberOfFolds =2;
end


 if isfield(options,'testIndex') || isfield(options,'trainIndex')
      numberOfFolds =1;  %user supplied test and training set
end

%This is for dimensionality reduction
if  isfield(options,'useDR')
    useDR=options.useDR;
    if isfield(options,'dim_reduction') % {'PCA', 'SLPP', 'SR', NPE'};dft 'SLPP'
        DR_ClassifyOptions.dim_reduction=options.dim_reduction;
    else
        DR_ClassifyOptions.dim_reduction='SLPP';
    end
     if isfield(options,'PCARatio') % set to <percent>, where percent is 0:1, dft 0.99
        DR_ClassifyOptions.PCARatio=options.PCARatio;
    else
        DR_ClassifyOptions.PCARatio=0.99;
     end
     if isfield(options,'dim_reduction_k')  %nearest neighbors, dft=0
        DR_ClassifyOptions.k=options.dim_reduction_k;
    else
        DR_ClassifyOptions.k=0;
     end
     if isfield(options,'SLPP_bLDA') % <alphaBelend>, where <alphaBlend> is 0:1
        %                                           0 is unsupervised, 1 is LDA supervised
        DR_ClassifyOptions.SLPP_bLDA=options.SLPP_bLDA;
    else
        DR_ClassifyOptions.SLPP_bLDA=0.5;
     end
    if isfield(options,'ReducedDim')  % <Max_d>, where <Max_d> is the maximum
        %                                          allowed output dimension, d
        DR_ClassifyOptions.ReducedDim=options.ReducedDim;
    else
        DR_ClassifyOptions.ReducedDim=99999;
    end
else
    useDR=0;  %don't use DR
end

modelData.method = method;

rng(2000);  %random number generator seed so results are repeatable
%Generate a fold value for each training sample
CVindex = crossvalind('Kfold',y, numberOfFolds);
i=1;  %this is for easier debugging....

for i = 1:numberOfFolds
    
    if (numberOfFolds == 1)  
        %user may specify either test or train split
        if isfield(options,'testIndex')
            TestIndex = options.testIndex;
            TrainIndex = find(~ismember(1:length(y),TestIndex));  %these need to be exclusive
        elseif isfield(options,'trainIndex')
            TrainIndex = options.trainIndex;
            TestIndex = find(~ismember(1:length(y),TrainIndex)); %these need to be exclusive
        else
            %user did not specify test or train split
            %then test = train set- be careful here!
            %Get train and test index values
            TestIndex = find(CVindex == i);
            TrainIndex = find(CVindex == i);
        end      
    else
        %Get train and test index values
        TestIndex = find(CVindex == i);
        TrainIndex = find(CVindex ~= i);
    end
    
    %for train and test partitions
    TrainXdata = Xdata(TrainIndex,:);
    TrainGT =y(TrainIndex);
    TestXdata = Xdata(TestIndex,:);
    TestGT = y(TestIndex);
    
	
    % Apply dimensionality reduction
    if useDR
        DRmat = getDRmat(TrainXdata',TrainGT,1,1,DR_ClassifyOptions);
        [bigD,littled]=size(DRmat);
        TrainXdata = TrainXdata*DRmat;   %[nxD]*[Dxd]  --> [nxd] top d dims  
        TestXdata = TestXdata*DRmat;   %[nxD]*[Dxd]  --> [nxd] top d dims  
        modelData.DRmat = DRmat;
    end
    
	
    %
    %build the model using TrainXdata and TrainGT
    %test the built model using TestXdata
    %
    switch method
        case 'LogisticRegression'
            
            if  isfield(options,'lambda')
                lambda = options.lambda;
            else
                lambda = 0;
            end
            
            % for Logistic Regression, we need to solve for theta
            % Initialize fitting parameters
            all_theta = zeros(numClasses, size(Xdata, 2));
            
            for c=1:numClasses
                % Set Initial theta
                initial_theta = zeros(size(Xdata, 2), 1);
                % Set options for fminunc
                opts = optimset('GradObj', 'on', 'MaxIter', 50);
                
                % Run fmincg to obtain the optimal theta
                % This function will return theta and the cost
                [theta] = ...
                    fmincg (@(t)(costFunctionLogisticRegression(t, TrainXdata, (TrainGT == c), lambda)), ...
                    initial_theta, opts);
                
                all_theta(c,:) = theta;
            end          
            
            % Using TestDataCV, compute testing set prediction using
            % the model created
            % for Logistic Regression, the model is theta
            all_pred = sigmoid(TestXdata*all_theta');
            [maxVal,maxIndex] = max(all_pred,[],2);
            TestDataPred=maxIndex;
			
			modelData.all_theta = all_theta;
            
        case 'KNN'
            if  isfield(options,'knn_k')
                knn_k = options.knn_k;
            else
                knn_k = 1;
            end
            [idx, dist] = knnsearch(TrainXdata,TestXdata,'k',knn_k);
            nnList=[];
            for i=1:knn_k
                nnList = [nnList TrainDataGT(idx(:,i))];
            end
            TestDataPred=mode(nnList')';
            
         case 'SVM'
             %Note- this is libsvm not the built-in svm functions to matlab
             if  isfield(options,'svm_t')
                svm_t = options.svm_t;
             else
                 svm_t = 0;
             end
             if  isfield(options,'svm_c')
                 svm_C = options.svm_c;
             else
                 svm_C = 1;
             end
             
             if  isfield(options,'svm_g')
                 svm_g = options.svm_g;
             else
                 svm_g = 1/n;
             end
             
             [TrainXdataNorm, mu, sigma] = featureNormalize(TrainXdata,0,0);
             eval(['model = svmtrain(TrainGT,TrainXdataNorm,''-t  ' num2str(svm_t)  ' -c ' num2str(svm_C) ' -g ' num2str(svm_g) ''' );']);
             
             [TestXdataNorm, mu, sigma] = featureNormalize(TestXdata,mu, sigma);
             TestDataPred = svmpredict( TestGT, TestXdataNorm, model, '-q');
                 
             modelData.mu = mu;
             modelData.sigma = sigma;
             modelData.model = model;
                 
         case 'ClassificationTree'
            tree = ClassificationTree.fit(TrainXdata,TrainGT);
            TestDataPred = predict(tree,TestXdata);
			
			modelData.tree = tree;
            
         case 'BaggedTree'
            rng(2000);  %random number generator seed
            t = ClassificationTree.template('MinLeaf',1);
            bagtree = fitensemble(TrainXdata,TrainGT,'Bag',10,t,'type','classification');
            TestDataPred = predict(bagtree,TestXdata);  %really should test with a test set here
			
			modelData.bagtree = bagtree;

        case 'Adaboost'
            if  isfield(options,'adaboost_numFeatures')
                adaboost_numFeatures = options.adaboost_numFeatures;
            else
                adaboost_numFeatures = round(n/2);
            end
            
            if numClasses ~= 2
                error('Adaboost only works with two class data');
            end
            %change class labels to -1 and +1
            yList = unique(TrainGT);
            if yList(1) ~= -1
                TrainGT(TrainGT==yList(1))=-1;
                TrainGT(TrainGT==yList(2))= 1;
                TestGT(TestGT==yList(1))=-1;
                TestGT(TestGT==yList(2))= 1;
            end
            
           [classifiers, errors,pred] = myAdaBoost(TrainXdata,TrainGT,adaboost_numFeatures,TestXdata,TestGT);
           allTrain(i,:) = errors.train;
           allTest(i,:) = errors.test;
           allEB(i,:) = errors.eb;
           
           TestDataPred = pred.test;
           TestDataPred(TestDataPred==1)=yList(2);
           TestDataPred(TestDataPred==-1)=yList(1);
           

        case 'nnet'
             if  isfield(options,'nnet_hiddenLayerSize')
                hiddenLayerSize=options.nnet_hiddenLayerSize;
            else
                hiddenLayerSize =10;
            end

            %Convert X and y data into Matlab nnet format:
            inputs = TrainXdata';
            
            %Convert to one-hot encoding ground truth values
            targets = zeros(C,length(TrainGT));
            for ii=1:length(TrainGT)
                targets(TrainGT(ii),ii) =  1;
            end
            
            % Create a Pattern Recognition Network
            setdemorandstream(2014784333);   %seed for random number generator
            net = patternnet(hiddenLayerSize);
            
            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 0.8;  %note- splits are done in a random fashion
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0.0;
            
            % Train the Network
            [net,tr] = train(net,inputs,targets);  %return neural net and a training record
            % plotperform(tr); %shows train, validation, and test per epoch
            
            %Convert X and y test data into Matlab nnet format:
            inputsTest = TestXdata';
            
            testY = net(inputsTest);   %pass all inputs through nnet
            TestDataPred=vec2ind(testY)';
            
            modelData.nnet = net;
			
			
        case 'SRC'
%             root = 'C:\Users\rwpeec\Desktop\rwpeec\research\compressedSensing\SLEP_4.0';
%             addpath(genpath([root '/SLEP']));
            dictionary = TrainXdata';   %[dxm]
            numDims = size(dictionary,1);
            zeroBasedClassIndex=0;
            if min(TrainGT) == 0  %if using zero based class indexx
                zeroBasedClassIndex=1;
                TrainGT = TrainGT+1;
                TestGT = TestGT+1;
            end
            numclasses = length(unique(TrainGT));
            
            % SR options
            lambda=0.15;
            opts=[];
            opts.init=2;        % starting from a zero point
            opts.tFlag=5;       % run .maxIter iterations
            opts.maxIter=100;   % maximum number of iterations
            opts.nFlag=0;       % without normalization
            opts.rFlag=0;       % the input parameter 'lambda' is a ratio in (0, 1)
            % note: by setting rFlag=1,lambda_max is automatically
            % computed, where lambda_max is the highest value for which
            % y=A*sols.  then lambda is multipled by this term to
            % regularlize l1(sols)
            opts.rsL2=0.5;      % the squared two norm term
            
            undefinedClass=numclasses+1;   %we will create an additional GT class for
            %situations where we are not sure how to
            %classify the sample
            
            clear TestDataPredict
            for yy = 1:length(TestGT)  %evaluate each test sample, one at a time
                %extract test sample
                 TestXdataSample = TestXdata(yy,:)';  %[dx1]
                
                %for each test sample, get sparse coeff's then use min reconstruction error to estimate class
                [sols, funVal1]= nnLeastR(dictionary, TestXdataSample, lambda, opts); %DictMHI is dxn, TestXdataSample is dx1
                %[sols, funVal1]= LeastR(dictionary, TestXdataSample, lambda, opts); %this does not enforce pos coefficients
                classOfTest = TestGT(yy);
                
                if length(unique(sols)) == 1
                    %all weights are equal, so assign to unknown class
                    TestDataPredict(yy) = undefinedClass;
                    %confMatrix(classOfTest,undefinedClass) = confMatrix(classOfTest,undefinedClass)+1;
                else
                    for cc = 1:numclasses  %%foreach class
                        classentries = find(TrainGT == cc); % find all training of each class
                        tmp =  find(sols(classentries) ~= 0);   % of those, which are non-zero
                        nonzeroentries = classentries(tmp);     % use only non-zero of each class
                        %If no coefficients, we guarantee large error, and
                        %thus that class will not be selected.
                        if length(nonzeroentries) == 0
                            classWeight(cc)=Inf;
                            
                        else
                            classEst=0;
                            for ss=1:length(nonzeroentries)
                                %A includes any dimensionality reduction, allxy3BigA_train does not
                                %classEst = classEst + sols(nonzeroentries(ss))*allxy3BigA_train(:,nonzeroentries(ss));
                                classEst = classEst + sols(nonzeroentries(ss))*dictionary(:,nonzeroentries(ss));
                            end
                            
                            %classWeight(cc) = sum(((classEst-allxy3BigA_test(:,yy)).^2)).^0.5; %Norm-2 error
                            classWeight(cc) = sum(((classEst-TestXdataSample).^2)).^0.5; %Norm-2 error
                        end
                    end
                    % award class to min reconstruction error
                    %classWeight
                    [minval,ind] = min(classWeight);
                    %classOfTest
                    classOfMaxSol = ind;
                    classificationConfidence1 = median(classWeight)-minval;  %the bigger the difference between minval and the median, the more confident we are
                    sortWeight = sort(classWeight);
                    classificationConfidence2 = (sortWeight(2)-sortWeight(1))/sortWeight(1);  %we simultaneously want large difference between closest two classes and a small min reconstruction error
                    %in this third method, we look at the ratio of minval over all the
                    %other  non-infinity values.  we multiply by the number of
                    %non-infinity values to get to a 0:1 scale
                    classificationConfidence3 = 1 - (sum(classWeight~=Inf)*minval / sum(classWeight(classWeight~=Inf)));
                    TestDataPredict(yy) = classOfMaxSol;
                    %confMatrix(classOfTest,classOfMaxSol) = confMatrix(classOfTest,classOfMaxSol)+1;
                end
            end
            if zeroBasedClassIndex == 1
                TestDataPredict = TestDataPredict-1;
            end
            TestDataPred=TestDataPredict';
            
        case 'KSVD'
            % Heavily modified code that that started with:
            %   Zhuolin Jiang, Zhe Lin, Larry S. Davis.
            %   "Learning A Discriminative Dictionary for Sparse Coding via Label
            %    Consistent K-SVD", CVPR 2011.
%             addpath C:\Users\rwpeec\Desktop\rwpeec\research\compressedSensing\KSVD\ksvdbox
%             addpath C:\Users\rwpeec\Desktop\rwpeec\research\compressedSensing\KSVD\ompbox
%             addpath C:\Users\rwpeec\Desktop\rwpeec\research\compressedSensing\KSVD
            
            % testing_feats: nxNtest input features, n dimensions, Ntest samples (Dxn in my termonology!)
            % training_feats: nxNtrain
            %We definitely don't want to do the next two lines...makes classification worse
            %training_feats = normcols(training_feats); %this is bad, leave commented out!
            %testing_feats = normcols(testing_feats);   %this is bad, leave commented out!
            training_feats = TrainXdata';  %dxn
            testing_feats = TestXdata';
            
            zeroBasedClassIndex=0;
            if min(TrainGT) == 0  %if using zero based class indexx
                zeroBasedClassIndex=1;
                TrainGT = TrainGT+1;
                TestGT = TestGT+1;
            end
            
            [dim,numTrainSamples] = size( training_feats);
            [dim,numTestSamples] = size( testing_feats);
            
            % H_test: cxNtest GT for test set, m classes, c is # classes, Ntest samples...
            %         one col per sample,row of class is 1, all other row entries are 0
            % H_train: cxNtrain GT for train set
            H_train = zeros(length(unique(TrainGT)),numTrainSamples);
            H_test = zeros(length(unique(TrainGT)),numTestSamples);
            for i=1:numTrainSamples
                H_train(TrainGT(i),i) = 1;
            end
            for i=1:numTestSamples
                H_test(TestGT(i),i) = 1;
            end
            
            
            sparsitythres = 30; % sparsity prior
            sqrt_alpha = 4; % weights for label constraint term
            sqrt_beta = 2; % weights for classification err term
            iterations = 50; % iteration number
            iterations4ini = 20; % iteration number for initialization
            
            %params.dictsize needs to be passed in.  If not passed in, make the size half the number of training samples.;
            if  isfield(options,'dictSize')
                dictsize=options.dictSize;
            else
                dictsize=round(numTrainSamples/2);
            end
            
            clear TestDataPredict
            switch options.KSVDmethod
                % run LC K-SVD Training (reconstruction err + class penalty)
                case 'do_KSVD'
                    %Note:  if useDR=0, this will give actual KSVD results
                    %generally expect better performance with useDR=1, as this
                    %will minimize coefficient contamination
                    
                    %[Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(training_feats,H_train,dictsize,iterations4ini,sparsitythres);
                     ClassifyOptions.ksvdVersion = 'use_v1';
                     dimReduc=0;
                     meanNorm=0;
                     [Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD_wDR_v2(training_feats,H_train,dictsize,iterations4ini,sparsitythres,dimReduc,meanNorm,ClassifyOptions);
                    %       Dinit           -initialized dictionary
                    %       Tinit           -initialized linear transform matrix  (matrix A in paper)
                    %       Winit           -initialized classifier parameters
                    %       Q               -optimal code matrix for training features
                    [prediction0,accuracy0] = classification(Dinit, Winit, testing_feats, H_test, sparsitythres);
                    TestDataPredict = prediction0;
                    
                case 'do_LCKSVD1'
                    fprintf('Dictionary learning by LC-KSVD1...');
                    [D1,X1,T1,W1] = labelconsistentksvd1(training_feats,Dinit,Q_train,Tinit,H_train,iterations,sparsitythres,sqrt_alpha);
                    [prediction1,accuracy1] = classification(D1, W1, testing_feats, H_test, sparsitythres);
                    TestDataPredict = prediction1;
                    
                case 'do_LCKSVD2'
                    % run LC k-svd training (reconstruction err + class penalty + classifier err)
                    fprintf('Dictionary and classifier learning by LC-KSVD2...')
                    [D2,X2,T2,W2] = labelconsistentksvd2(training_feats,Dinit,Q_train,Tinit,H_train,Winit,iterations,sparsitythres,sqrt_alpha,sqrt_beta);
                    [prediction2,accuracy2] = classification(D2, W2, testing_feats, H_test, sparsitythres);
                    TestDataPredict = prediction2;
                    
                case 'do_LGEKSVD'
                    if useDR
                        error('The point of this method is to solve simultaneously for the DRmat along with the dictionary')
                    end
                    
                    ksvdparams.FG13_paper=1;   %use the Face and Gesture K-LGE, unmodified ksvd function, ksvd.m
                    ksvdparams.CVPR13_paper=0; %use the CVPR LGE-KSVD version of ksvd, ksvd_wLGE.m
                    
                    dimReduc=1;
                    meanNorm=0;   %0 means skip mean normalization of data
                    
                    %need to pick one of these dim reduction techniques
                    %ClassifyOptions.dim_reduction='PCA'; ClassifyOptions.PCARatio=0.9999;
                    ClassifyOptions.dim_reduction='SLPP'; ClassifyOptions.k = 5; ClassifyOptions.SLPP_bLDA = 0.5; ClassifyOptions.PCARatio=0.9999; ClassifyOptions.ReducedDim=5000;
                    %ClassifyOptions.dim_reduction='SR'; ClassifyOptions.ridge = 1; ClassifyOptions.ReguAlpha=0.15; ClassifyOptions.LassoCardi=500;
                    %ClassifyOptions.dim_reduction='SR'; ClassifyOptions.ridge = 0; ClassifyOptions.ReguAlpha=1; ClassifyOptions.LassoCardi=500;
                    %ClassifyOptions.dim_reduction='NPE'; ClassifyOptions.k = 5; ClassifyOptions.PCARatio=0.9999; ClassifyOptions.ReducedDim=510;
                    
                    ClassifyOptions.t = 1; % this was used for F&G and CVPR
                    %ClassifyOptions.t = 10000;  %this does a better job for unsupervised portion of Wadj on some datasets
                    
                    %ClassifyOptions.LearningParameter=1;  %in initialization4LCKSVD_wDR, what portion of new DRmat to use
                    ClassifyOptions.LearningParameter=0.25; %in initialization4LCKSVD_wDR, what portion of new DRmat to use
                    ClassifyOptions.DRsolve = 'Direct';  %in initialization4LCKSVD_wDR, solves for new DRmat via pinv()
                    %ClassifyOptions.DRsolve = 'SPP_S';   %in initialization4LCKSVD_wDR, solves for new DRmat via sparstity preserving projections, use coeffs directly as W matrix
                    %ClassifyOptions.DRsolve = 'SPP_Sb';  %in initialization4LCKSVD_wDR, solves for new DRmat via sparstity preserving projections, use C+C'-C'C as W matrix
                    ClassifyOptions.gammamode = 'use_SPARESECODE'; %default ksvd, +/- coeffs
                    %ClassifyOptions.gammamode = 'use_LeastR'; %SLEP sparse coeff generator
                    %ClassifyOptions.gammamode = 'use_nnLeastR'; %SLEP coeff generator, all nn coeffs
                    ClassifyOptions.lambda = 0.15;  %lambda regularizer for SLEP
                    
                    maxDRmat_iter=100;   %max number of iterations in KLGE
                    updateAtomList={'ksvd_LGE_minE'};  %this is CVPR and TIP paper
                    ksvdparams.importantCoeffThreshold=0.5; %when picking out peaks from coefficients, select all peaks whos percent entergy is greater than this
                    
                    ksvdparams.AtomGroup='initAssigned';  %default, atom keeps initial assigment
                    %ksvdparams.AtomGroup='maxPeak'; %atom takes on class associated with max coefficient
                    %ksvdparams.AtomGroup='maxEnergy'; %atom takes on class associated with max energy across all coefficients grouped by class
                    
                    ksvdparams.LGEWadjmethod='modify';
                    ksvdparams.NeighborMode = 'KNN';
                    ksvdparams.k = 5;  %0 is all
                    
                    importantCoeffThresholdList=[0 0.1 0.25 0.5 0.75 0.9 1];
                    coeffCounter=4;
                    ksvdparams.importantCoeffThreshold = importantCoeffThresholdList(coeffCounter);
                    
                    % initialization4LCKSVD_rwp is the same as initialization4LCKSVD, except
                    % rather than have the same number of dictionary entries per class, we
                    % instead mimic the proportion of training samples in the dictionary.  So,
                    % if one class has many more training samples than another, it would have
                    % more dictionary elements.
                    ClassifyOptions.ksvdVersion = 'use_v2';
                    ksvdparams.FG13_paper=1;  %ksvdparams.CVPR13_paper=0;
                    dimReduc=1;
                    
                    %[DinitDR,TinitDR,WinitDR,Q_trainDR,DRmat,Wadj] = initialization4LCKSVD_wDR_v2(training_feats_D,H_train,dictsize,iterations4ini,sparsitythres,dimReduc,meanNorm,ClassifyOptions);
                    %[prediction0dr2,accuracy0dr2] = classification(DinitDR, WinitDR, DR_test, H_test, sparsitythres);
                    [Dbest,Xbest,Cbest,DRmatbest,Tinit,Q,Wadj,running_accuracy,Cnext_lib_SVMbest,running_accuracySVM] = initialization4LCKSVD_KLGE2(training_feats,H_train,testing_feats,H_test,dictsize,iterations4ini,sparsitythres,dimReduc,maxDRmat_iter,ClassifyOptions,ksvdparams);
                    [prediction0dr2,accuracy0dr2,err0dr2,predSVM0dr2,accSVM0dr2,errSVM0dr2] = classification(Dbest, Cbest, DR_test, H_test, sparsitythres,Cnext_lib_SVMbest);
                    
                    TestDataPredict = prediction0dr2;
                    
                otherwise
                    error('Unknown options.KSVDmethod')
            end
            
            if zeroBasedClassIndex == 1
                TestDataPredict = TestDataPredict-1;
            end
            TestDataPred=TestDataPredict';
            
            
        otherwise
            error('Unknown classification method')
    end
    
    predictionLabels(TestIndex,:) =double(TestDataPred);
end

if (numberOfFolds == 1)  && ( isfield(options,'testIndex') || isfield(options,'trainIndex'))
    %if there are pre-defined test and training sets, compute confusion
    %matrix only on test set
    confusionMatrix = confusionmat(y(TestIndex),predictionLabels(TestIndex));
else
    %compute confusion matrix using all samples
    confusionMatrix = confusionmat(y,predictionLabels);
end
accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));

if useDR
        fprintf(sprintf('%s w/DR, D= %d, d= %d: Accuracy = %6.2f%%%% \n',method,bigD,littled,accuracy*100));
else
    fprintf(sprintf('%s: Accuracy = %6.2f%%%% \n',method,accuracy*100));
end
fprintf('Confusion Matrix:\n');
[r c] = size(confusionMatrix);
for i=1:r
    for j=1:r
        fprintf('%6d ',confusionMatrix(i,j));
    end
    fprintf('\n');
end

if strcmp(options.method,'Adaboost')
    meanTest = mean(allTest);
    meanTrain = mean(allTrain);
    meanEB  = mean(allEB);
    figure
    hold on
    x = 1:1:adaboost_numFeatures;
    plot(x, meanEB,'k:',x,meanTest,'r-',x,meanTrain,'b--','LineWidth',2);
    legend('ErrorBound','TestErr','TrainErr','Location','Best');
    xlabel 'iteration (number of Classifiers)'
    ylabel 'error rates (50 trials)'
    title 'AdaBoost Performance on Bupa'
    %print -dpng hwk5_11.png
end