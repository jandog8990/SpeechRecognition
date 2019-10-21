[y1, Fs1] = audioread('test/s4.wav');
% sample = y1[1:

% pull 3.5 sec of data from signal
si = 3.5*Fs1;
s = y1(8224:8224+10);
plot(y1);
sound(y1, Fs1)