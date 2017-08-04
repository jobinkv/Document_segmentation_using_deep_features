clc; clear all; close all;
% Load data
% load '../mnist/mnist_train.mat'
% ind = randperm(size(train_X, 1));
% train_X = train_X(ind(1:5000),:);
% train_labels = train_labels(ind(1:5000));
% % Set parameters
no_dims = 2;
initial_dims = 400;
perplexity = 30;
disp('yahoooo')
% % Run t?SNE
% mappedX = tsne(train_X, [], no_dims, initial_dims, perplexity);
% Plot results
 load('tsneData.mat');
 mappedX = tsne(tsneData.feature, [], no_dims, initial_dims, perplexity);
save('mappedX.mat','mappedX');
keyboard
% load('/Users/jobinkv/tsneLabels.mat');
gscatter(mappedX(:,1), mappedX(:,2), tsneData.Labels');
