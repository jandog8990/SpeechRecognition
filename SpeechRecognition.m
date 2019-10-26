[y1, Fs1] = audioread('test/s4.wav');
% sample = y1[1:

% pull 3.5 sec of data from signal
si = 3.5*Fs1;
s = y1(8224:8224+10);
figure
plot(y1);
sound(y1, Fs1)

% Plot Mel Frequency Cepstrum Coeff triangulatr magnitude
figure
plot(linspace(0, (12500/2), 129), melfb(20, 256, 12500)'),
title('Mel-spaced filterbank'), xlabel('Frequency (Hz)');