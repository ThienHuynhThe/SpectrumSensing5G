function [imdsTrain,pxdsTrain,imdsVal,pxdsVal] = ...
  helperSpecSensePartitionData(imds,pxds,parts)
%helperSpecSensePartitionData Partition training data
%   [IDSTR,PDSTR,IDSVL,PDSVL]=helperSpecSensePartitionData(IDS,PDS,P)
%   paritions the image data store, IDS, and pixel label data store, PDS,
%   into training image data store, IDSTR, training pixel data store,
%   PDSTR, validation image data store, IDSVL, and validation pixel label
%   data store, PDSVL, based on the paritioning percentatges set in P,
%   P is a 1x2 vector with sum of the elements equal to 100.

%   Copyright 2021 The MathWorks, Inc.

assert(sum(parts)==100, 'Sum of parts must be 100')

% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

numTrain = round(numFiles*parts(1)/100);

imdsTrain = subset(imds, shuffledIndices(1:numTrain));
imdsVal = subset(imds, shuffledIndices(numTrain+1:end));

pxdsTrain = subset(pxds, shuffledIndices(1:numTrain));
pxdsVal = subset(pxds, shuffledIndices(numTrain+1:end));

end