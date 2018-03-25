%Prepare the imdb structure, returns image data with mean image subtracted

function imdb = getMnistImdb(opts)
load('ex4data1.mat');

set = [ones(1,4000) 3*ones(1,1000)];
idx = randperm(size(X,1));
X=X(idx,:);
y=y(idx);

data = permute(single(reshape(X,[],20,20,1)),[2 3 4 1]);
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

%split into train and test data
[trainInd,ValInd, testInd] = divideblock(5000,0.8,0.1,0.1);

imdb.images.y = y;
imdb.images.data_train = data(:,:,:,trainInd);

testInd = cat(2,testInd,ValInd);
imdb.images.data_test = data(:,:,:,testInd);
imdb.images.data_mean = dataMean;

imdb.images.labels_train_gt = categorical(y(trainInd))';
imdb.images.labels_test_gt = categorical(y(testInd));
end

