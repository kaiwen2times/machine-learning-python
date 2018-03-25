%% Transfer Learning Using AlexNet
% This example shows how to fine-tune a pretrained AlexNet convolutional
% neural network to perform classification on a new collection of images..
% Adapted https://www.mathworks.com/help/nnet/examples/transfer-learning-using-alexnet.html
% Prof. Ray Ptucha, RIT 2017
clear; clc;
%% Load Data
% Unzip and load the new images as an image datastore.
% |imageDatastore| automatically labels the images based on folder names
% and stores the data as an |ImageDatastore| object and
% efficiently read batches of images during training of a convolutional
% neural network.
unzip('MerchData.zip');
images = imageDatastore('MerchData',...
                        'IncludeSubfolders',true,...
                        'LabelSource','foldernames');

%%
% Divide the data into training and validation data sets. Use 70% of the
% images for training and 30% for validation. |splitEachLabel| splits the
% |images| datastore into two new datastores.
[trainingImages,validationImages] = splitEachLabel(images,0.7,'randomized');

%%
% This very small data set now contains 55 training images and 20
% validation images. Display some sample images.
numTrainImages = numel(trainingImages.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
subplot(4,4,i)
I = readimage(trainingImages,idx(i));
imshow(I)
end

%% Load Pretrained Network
% Load the pretrained AlexNet neural network. If Neural Network Toolbox(TM)
% Model _for AlexNet Network_ is not installed, then the software provides
% a download link. AlexNet is trained on more than one million images and
% can classify images into 1000 object categories, such as keyboard, mouse,
% pencil, and many animals. As a result, the model has learned rich feature
% representations for a wide range of images.
net = alexnet;

%%
% Display the network architecture. The network has five convolutional
% layers and three fully connected layers.
net.Layers

%layers_alexnet = net.Layers;
%disp('Generating plot before training..')
%figure
%lgraph = layerGraph(layers_alexnet);
%plot(lgraph)
%title('Before replacing the layers.')

%% Transfer Layers to New Network
% The last three layers of the pretrained network |net| are configured for
% 1000 classes. These three layers must be fine-tuned for the new
% classification problem. Extract all layers, except the last three, from
% the pretrained network.
layersTransfer = net.Layers(1:end-3);

%%
% Transfer the layers to the new classification task by replacing the last
% three layers with a fully connected layer, a softmax layer, and a
% classification output layer. Specify the options of the new fully
% connected layer according to the new data. Set the fully connected layer
% to have the same size as the number of classes in the new data. To learn
% faster in the new layers than in the transferred layers, increase the
% |WeightLearnRateFactor| and |BiasLearnRateFactor| values of the fully
% connected layer.

numClasses = numel(categories(trainingImages.Labels))
layers = [
          layersTransfer
          fullyConnectedLayer(numClasses,'Name','fc',...
                              'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
          softmaxLayer
          classificationLayer];



%% Train Network
% Specify the training options.
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
                          'MiniBatchSize',miniBatchSize,...
                          'MaxEpochs',4,...
                          'InitialLearnRate',1e-4,...
                          'Verbose',true,...
                          'Plots','training-progress',...
                          'ValidationData',validationImages,...
                          'ValidationFrequency',numIterationsPerEpoch);

%%
% Train the network that consists of the transferred and new layers.
if exist('netTransfer.mat', 'file')
disp('Loading saved model..')
load netTransfer;
else
%Train the network.
netTransfer = trainNetwork(trainingImages,layers,options);
save netTransfer;
end

%disp('Generating plot after training..')
%layers = netTransfer.Layers;
%lgraph = layerGraph(layers);
%figure
%plot(lgraph)
%title('After replacement of layers and training...')

%% Classify Validation Images
% Classify the validation images using the fine-tuned network.
predictedLabels = classify(netTransfer,validationImages);

%%
% Calculate the classification accuracy on the validation set. Accuracy is
% the fraction of labels that the network predicts correctly.
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels)

