% ========================================================================
% Initialization for Label consistent KSVD algorithm
% USAGE: [DRmat,Wadj] = getDRmat(training_feats,H_train,dimReduc,meanNorm,ClassifyOptions)
% Inputs
%       training_feats  -training features, Dxn
%       H_train         -label matrix for training feature 
%       dimReduc        -0 no DR, 1 use DR
%       meanNorm        -not used anymore...here for backward compatibility
%       ClassifyOptions -
%                        ClassifyOptions.dim_reduction={'PCA', 'SLPP', 'SR', NPE'}
%                        ClassifyOptions.PCARatio=<percent>, where percent is 0:1, dft 0.99
%                        ClassifyOptions.k = <numNearestNeighbors>
%                        ClassifyOptions.SLPP_bLDA = <alphaBelend>, where <alphaBlend> is 0:1
%                                                    0 is unsupervised, 1 is LDA supervised
%                        ClassifyOptions.ReducedDim = <Max_d>, where <Max_d> is the maximum
%                                                    allowed output dimension, d
%                        if dim_reduction='SR', we additionally have: 
%                           ClassifyOptions.ridge = 0 for ridge, 1 for lasso
%                           ClassifyOptions.ReguAlpha = regularization value 
%                           ClassifyOptions.LassoCardi = LassoCardi value
%
% Outputs
%       DRmat           -dimensionality reduction matrix, Dxd
%                        Sample usage:
%                        DR_train = training_feats'*DRmat;  %[nxD]*[Dxd]  --> [nxd]
%                        DR_train = DR_train'; % DR_train is now [dxn]
%       Wadj            -adjacency matrix W from training_feats
%   
% Author: Ray Ptucha (rwpeec@rit.edu)
% Date: 08-17-2012
% 
% Updates:
% Date   Person     Change
% ========================================================================
function [DRmat,Wadj] = getDRmat(training_feats,H_train,dimReduc,meanNorm,ClassifyOptions)

%Want GTtrain=1x<num_train_samples>
[r,c] = size(H_train);
if (r==1 || c==1)
    GTtrain=H_train;
    if c == 1
        GTtrain=GTtrain';
    end
else
    %convert from nnet style GT to vector GT
    GTtrain=zeros(1,size(H_train,2));  
    for t=1:size(H_train,2)
        GTtrain(t) = find(H_train(:,t) == 1);
    end
end
    
if (dimReduc)    
    
    if (strcmp(ClassifyOptions.dim_reduction,'PCA'))
        fprintf('Doing PCA...\n');
        if (0)
            PCA_fea_train = training_feats';  %PCA_fea_train = <num_train_samples> x <D>
            %PCA_fea_test = testing_feats';  %PCA_fea_test = <num_train_samples> x <D>

            cov_mat_fea_train = cov(PCA_fea_train);
            % eigvector = <DxD>, eigvalue = <DxD>
            [PCA_eigvector, PCA_eigvalue] = eig(cov_mat_fea_train);
            % eigvalue = <Dx1>
            PCA_eigvalue = diag(PCA_eigvalue);

            %sort so most important is top to bottom, left to right
            PCA_eigvalue = flipud(PCA_eigvalue);
            PCA_eigvector = fliplr(PCA_eigvector);
        else
            if isfield(ClassifyOptions,'ReducedDim')
                saveReducedDim=ClassifyOptions.ReducedDim;
            end
            clear options
            options.ReducedDim=0;  %keep all dimensions
            [PCA_eigvector, PCA_eigvalue] = PCA_Cai(training_feats', options);
        end

        sum_eig = sum(PCA_eigvalue);
        for ev=1:length(PCA_eigvalue)
            running_sum = sum(PCA_eigvalue(1:ev));
            if running_sum/sum_eig > ClassifyOptions.PCARatio
                break;
            end
        end
        maxPCA=ev;
        disp(sprintf('Number of dimensions, d = %d\n',maxPCA));
        
        if saveReducedDim < maxPCA
            maxPCA = saveReducedDim;
            disp(sprintf('Number of dimensions trimmed by ReducedDim option, d = %d\n',maxPCA));
        end
        DRmat = PCA_eigvector(:,1:maxPCA);  %Dxd

        % Now apply LPP eig matrix on data
%         PCA_output = PCA_fea_train*PCA_eigvector(:,1:maxPCA);   %[nxD]*[Dxd]  --> [nxd]
%         training_feats = PCA_output';
%         PCA_output = PCA_fea_test*PCA_eigvector(:,1:maxPCA);   %[nxD]*[Dxd]  --> [nxd]
%         testing_feats = PCA_output';

        
            
            
    elseif (strcmp(ClassifyOptions.dim_reduction,'SLPP'))
        fprintf('Doing SLPP...\n');
        SLPP_fea_train = training_feats';  %SLPP_fea_train = <num_train_samples> x <D>
        %SLPP_fea_test = testing_feats';  %SLPP_fea_test = <num_train_samples> x <D>

        options=[];
        if isfield(ClassifyOptions,'t')
            options.t = ClassifyOptions.t;
        else
            options.t = 1;
        end
        options.Metric = 'Euclidean';
        options.NeighborMode = 'Supervised';
        
        options.gnd = GTtrain;          %gnd = 1x<num_train_samples>    
        % W is the Laplacian matrix:
        % In supervised mode, this is from ground truth
        % In unsupervised mode, this is from neighbor distances
        if ClassifyOptions.SLPP_bLDA == 1
            options.bLDA = 1;  %this was 1 orig for first year of using this
            W = constructW(SLPP_fea_train,options);  % W is <num_train_samples> x <num_train_samples>
        elseif ClassifyOptions.SLPP_bLDA == 0
            options.bLDA = 0;  
            %options.k = ClassifyOptions.k; %k=0: put and edge if from same class; k>0: put and edge if from same class and among k nearest neighbors
            options.WeightMode = 'HeatKernel';  
            W = constructW(SLPP_fea_train,options);  % W is <num_train_samples> x <num_train_samples>
        else
            options.bLDA = 0;  
            options.WeightMode = 'HeatKernel';  
            W1 = constructW(SLPP_fea_train,options);  % W is <num_train_samples> x <num_train_samples>
            options.bLDA = 1;
            W2 = constructW(SLPP_fea_train,options);
            ClassifyOptions.SLPP_bLDA_mixmode='onlySameClassNonZeroEntries';
            if strcmp(ClassifyOptions.SLPP_bLDA_mixmode,'noEntryRestriction')
                W = (1-ClassifyOptions.SLPP_bLDA)*W1 + ClassifyOptions.SLPP_bLDA*W2;
            elseif strcmp(ClassifyOptions.SLPP_bLDA_mixmode,'onlySameClassNonZeroEntries')
                %spones(W2) replace non-zeros in W2 with ones
                W1 = W1.*spones(W2);     %constrain W1 to only connect samples in same class
                W = (1-ClassifyOptions.SLPP_bLDA)*W1 + ClassifyOptions.SLPP_bLDA*W2;
            else
                display('Error, ClassifyOptions.SLPP_bLDA_mixmode need to be noEntryRestriction or onlySameClassNonZeroEntries');
            end
        end
        options.PCARatio = ClassifyOptions.PCARatio; %percent of data kept after PCA
        options.ReducedDim = ClassifyOptions.ReducedDim; %max dim of reduced space, (if 0 all kept)*, dft=30   *might not work
        [LPP_eigvector, LPP_eigvalue] = lpp(W, options, SLPP_fea_train); % eigvector = <Dxd>, eigvalue = <dx1>
        disp(sprintf('Number of dimensions, d = %d\n',length(LPP_eigvalue)));
        %LPP_eigvalue(end);

        DRmat = LPP_eigvector;  %Dxd
        Wadj=W; %nxn
        
        % Now apply LPP eig matrix on data
%         LPP_output = SLPP_fea_train*LPP_eigvector;  %[nxD]*[Dxd]  --> [nxd]
%         training_feats = LPP_output';
%         LPP_output = SLPP_fea_test*LPP_eigvector;  %[nxD]*[Dxd]  --> [nxd]
%         testing_feats = LPP_output';
        
    elseif (strcmp(ClassifyOptions.dim_reduction,'SR'))
        fprintf('Doing SR (Spectral Regression)...\n');
        SR_fea_train = training_feats';  %SLPP_fea_train = <num_train_samples> x <D>
        %SR_fea_test = testing_feats';  %SLPP_fea_test = <num_train_samples> x <D>
        
        options=[];
        options.gnd = GTtrain;          %gnd = 1x<num_train_samples>    
        
        if ClassifyOptions.ridge
            options.ReguAlpha=ClassifyOptions.ReguAlpha;  %0.01 good default
            options.ReguType='Ridge';
            [SR_eigvector] = SR_caller(options, SR_fea_train); % eigvector = <Dxd>, eigvalue = <dx1>

            DRmat = SR_eigvector;  %Dxd
            % Now apply LPP eig matrix on data
%             SR_output = SR_fea_train*SR_eigvector;
%             SR_output_test = SR_fea_test*SR_eigvector;
        else
            options.ReguAlpha=ClassifyOptions.ReguAlpha;  %0.001 good default
            options.ReguType='RidgeLasso';
            %options.LassoCardi = [10:5:60]; %the problem with this is that it will create set of eigs, one for each value here
            options.LassoCardi = ClassifyOptions.LassoCardi;  %90 good value here
            [SR_eigvectorALL] = SR_caller(options, SR_fea_train); % eigvector = <Dxd>, eigvalue = <dx1>

            SR_eigvector=SR_eigvectorALL{1};
            DRmat = SR_eigvector;  %Dxd
            % Now apply LPP eig matrix on data
%             SR_output = SR_fea_train*SR_eigvector;
%             SR_output_test = SR_fea_test*SR_eigvector;
        end

%         training_feats = SR_output';
%         testing_feats = SR_output_test';
        
    elseif (strcmp(ClassifyOptions.dim_reduction,'NPE'))
        fprintf('Doing NPE...\n');
        NPE_fea_train = training_feats';  %NPE_fea_train = <num_train_samples> x <D>
        %NPE_fea_test = testing_feats';  %NPE_fea_test = <num_train_samples> x <D>

        options=[];
        options.NeighborMode = 'Supervised';
        options.k = ClassifyOptions.k;
        
        options.PCARatio = ClassifyOptions.PCARatio; %percent of data kept after PCA
        options.ReducedDim = ClassifyOptions.ReducedDim; %max dim of reduced space, (if 0 all kept)*, dft=30   *might not work
        
        options.gnd = GTtrain;          %gnd = 1x<num_train_samples>    
        
        [NPE_eigvector, NPE_eigvalue] = NPE(options, NPE_fea_train); % eigvector = <Dxd>, eigvalue = <dx1>
        disp(sprintf('Number of dimensions, d = %d\n',length(NPE_eigvalue)));
        NPE_eigvalue(end)

        DRmat = NPE_eigvector;  %Dxd
        
        % Now apply NPE eig matrix on data
%         NPE_output = NPE_fea_train*NPE_eigvector;  %[nxD]*[Dxd]  --> [nxd]
%         training_feats = NPE_output';
%         NPE_output = NPE_fea_test*NPE_eigvector;  %[nxD]*[Dxd]  --> [nxd]
%         testing_feats = NPE_output';
    end
    
else
    DRmat = eye(size(training_feats,1),size(training_feats,1));
end

%if we didn't call SLPP dim reduction, we need to calculate W
if ~dimReduc || ~(strcmp(ClassifyOptions.dim_reduction,'SLPP'))
    options=[];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'Supervised';
    
    options.gnd = GTtrain;          %gnd = 1x<num_train_samples>    
    % W is the Laplacian matrix:
    % In supervised mode, this is from ground truth
    % In unsupervised mode, this is from neighbor distances
    options.bLDA = 1;  %this was 1 orig for first year of using this
    Wadj = constructW(training_feats',options);  % W is <num_train_samples> x <num_train_samples>
end