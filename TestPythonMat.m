% Test output *.mat files from Python
clear all
close all
clc
traindir = './train/s';
N = 6;
k = 16;
MFCC = {};
code = {};
names = [""];
for i = 1:N
    name = "s" + string(i);
    file = traindir + string(i) + "_mfcc.mat";
    disp(file);
    mfcc = load(file);
    MFCC{i} = mfcc.mfcc;
    names(i) = name;
    
%     s{i} = load(file);
%     code{i} = vqlbg(mfcc, k);
end
for i = 1:N
    code{i} = vqlbg(MFCC{i}, k);
end
disp("Codes:");
disp(code);
disp("\n");

% Run a test on new code using codes and test files
testdir = "./test/s";
test(testdir, N, code);

