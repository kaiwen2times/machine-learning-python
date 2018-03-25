% CNN_MNIST   Demonstrates  CNN using Neural Net Toolbox on MNIST 5000
% Adapted from https://www.mathworks.com/help/nnet/examples/create-simple-deep-learning-network-for-classification.html
% Prof. Ray Ptucha, RIT 2017

clear;
clc;

rng(1000);
opts.train.learningRate = 0.001 ;
opts.train.numEpochs = 1;
opts.dataDir = fullfile('./') ;  %note: for windows change it to fullfile('.\')
opts.imdbPath = fullfile(opts.dataDir, 'ex4data1_lmdb');
opts.train.batchSize = 10 ;
opts.train.expDir = opts.dataDir ;

%Prepare data and model
if exist('ex4data1_lmdb.mat', 'file')
    imdb = load('ex4data1_lmdb.mat') ;
else
    imdb = getMnistImdb(opts) ;
    if (~exist(opts.dataDir,'dir'))
          mkdir(opts.dataDir) ;
    end
    %save the processed data as a structure
    save(opts.imdbPath, '-struct', 'imdb');
end

%calls the function that defines the convolutional neural network architecture.

[layers, options] = cnn_architecture(opts);

%if exist('convnet.mat', 'file')
%    load convnet;
%else
%Train the network.
[convnet, info]= trainNetwork(...
                imdb.images.data_train,...
                imdb.images.labels_train_gt,...
                layers,options);
%save convnet;
%end
%Run the trained network on the test set that was not used to train the network and predict the image labels (digits).
YTest = classify(convnet,imdb.images.data_test);

%Calculate the accuracy.
accuracy = sum(YTest == imdb.images.labels_test_gt)/numel(imdb.images.labels_test_gt)


