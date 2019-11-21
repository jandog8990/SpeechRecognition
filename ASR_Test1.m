% Run the training and testing on acoustic vectors
clear all;
close all;
clc;
n = 6;
traindir = './train';
testdir = './test';
code = train(traindir, n);
test(testdir, n, code);