% ----------------------------------------
% Demo the input speaker audio samples
% ----------------------------------------
close all
clear all
clc
Fs = 44100;
bits = 16;
numChan = 1;
audioInfo = audiodevinfo;       % your machines info
inputID = audioInfo.input.ID; % ID of your input
recorder = audiorecorder(Fs, bits, numChan, inputID);
record(recorder)
disp("Please talk for 5 sec...")
pause(8)
stop(recorder)
play(recorder)
y = getaudiodata(recorder);

% Plot input signal
figure()
plot(y)
title("Speaker Input Test:")
hold on

% Save signal to file
% close all
filename = 'speakerTest.wav';
audiowrite(filename, y, Fs);

% Replay the saved wav sample
% clear y Fs
% [y, Fs] = audioread(filename);
% pause(3)
% sound(y,Fs);
