clear
clc
% Load input data
% please update the folder of training input data
imds = imageDatastore('...\data\trainingSet\input','IncludeSubfolders',false,'FileExtensions','.png');
% Define three classes 
classNames = ["NR" "LTE" "Noise"];
pixelLabelID = [127 255 0];
% Load groundtruth data
% please update the folder of training groundtruth data
pxdsTruth = pixelLabelDatastore('...\data\trainingSet\label',classNames,pixelLabelID,...
  'IncludeSubfolders',false,'FileExtensions','.png');
% Analyze Dataset Statistics
tbl = countEachLabel(pxdsTruth);
frequency = tbl.PixelCount/sum(tbl.PixelCount);
figure
bar(1:numel(classNames),frequency)
grid on
xticks(1:numel(classNames)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%Prepare Training and Validation
[imdsTrain,pxdsTrain,imdsVal,pxdsVal] = helperSpecSensePartitionData(imds,pxdsTruth,[80 20]);
cdsTrain = combine(imdsTrain,pxdsTrain);
cdsVal = combine(imdsVal,pxdsVal);

% Apply a transform to resize the image and pixel label data
imageSize = [256 256];
cdsTrain = transform(cdsTrain, @(data)preprocessTrainingData(data,imageSize));
cdsVal = transform(cdsVal, @(data)preprocessTrainingData(data,imageSize));

% Load the architecture of deep model
load('SRNet.mat');
% Balance classes using class weighting
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"labels",pxLayer);


%Select training options
opts = trainingOptions("sgdm",...
  MiniBatchSize = 40,...
  MaxEpochs = 100, ...
  LearnRateSchedule = "piecewise",...
  InitialLearnRate = 0.02,...
  LearnRateDropPeriod = 10,...
  LearnRateDropFactor = 0.1,...
  ValidationFrequency = 200,...
  ValidationData = cdsVal,...
  ValidationPatience = inf,...
  BatchNormalizationStatistics="moving",...
  Shuffle="every-epoch",...
  OutputNetwork = "best-validation-loss",...
  Plots = 'training-progress');

[net,trainInfo] = trainNetwork(cdsTrain,lgraph,opts); 

% Performance evaluation on test data
imds = imageDatastore('...\data\testSet\input' ,'IncludeSubfolders',false,'FileExtensions','.png');
pxdsResults = semanticseg(imds,net,"WriteLocation",'...\data\testSet\output'); % change the location to save segmented outputs 

% Evaluate
pxdsTruth = pixelLabelDatastore('...\data\testSet\label',classNames,pixelLabelID,...
  'IncludeSubfolders',false,'FileExtensions','.png');
% Measure the accuracy, IoU, and other metrics
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);
