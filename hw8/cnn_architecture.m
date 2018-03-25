%Define the convolutional neural network architecture.
function [layers, options] = cnn_architecture(opts)

layers = [imageInputLayer([20 20 1]);
          convolution2dLayer(5,20,'Stride',1,'Padding',2);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(10);
          softmaxLayer();
          classificationLayer()];
        
      
%Set the options to default settings for the stochastic gradient descent with momentum. Set the maximum number of epochs at 1, and start the training with an initial learning rate of 0.001.
options = trainingOptions('sgdm','MaxEpochs',opts.train.numEpochs,...
            'InitialLearnRate',opts.train.learningRate, ...
            'MiniBatchSize',opts.train.batchSize);