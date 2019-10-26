function c = mfcc(s, fs)
% MFCC Calculate the mel frequencey cepstrum coefficients (MFCC) of a signal
% Inputs:
%       s       : speech signal
%       fs      : sample rate in Hz
% Outputs:
%       c       : MFCC output, each column contains the MFCC's for one speech frame

N = 256;
M=100;
win_frames = zeros(N , floor((length(s)-N+M)/M-1));
frames=zeros(N , floor((length(s)-N+M)/M-1));

n=[0:N-1]';
w_n = 0.54 - 0.46*cos(2*pi*n/(N-1));

for i = 0 : floor((length(s)-N+M)/M-1)
    %the columns of frames are the blocked signal pieces.
    frame = s( i*M+1 : i*M+1 + N -1);
    
%     Can use the below two lines to plot the image of the frame blocked signal
%     Can use the same imagesc to show signal after Hamming windowing (can
%     see the signal tapering off in each signal block)
     frames(:,i+1) = s( i*M+1 : i*M+1 + N -1);
%     imagesc(frames)

    %applys the Hamming windowing to the framed signal;
    win_frames(: , i+1) = frame.*w_n;

end

fft_frames = zeros(size(win_frames));
%for loop computes the fft of the Hamming windowed frames. 
for i = 1 : size(win_frames , 2)
   fft_frames(:,i) = fft(win_frames(:,i)); 
   %imagesc(abs(fft_frames))
end

%Determine matrix for a mel-spaced filterbank
 m = melfb(20, N, fs);
 N2 = 1 + floor(N/2);
Sk=zeros(size(m,1), size(fft_frames,2));
%to plot mel filterbanks:
% plot(linspace(0, (12500/2), 129), melfb(20, 256, 12500)')

%Compute the mel-scale spectrum of the FFT frames
for i = 1 : size(fft_frames,2)
 Sk(:,i) = m * abs(fft_frames(1:N2,i)).^2;
 %to plot mel frequency spectrum: imagesc(Sk)
end

ms = Sk;
c = [];
for i = 1 : size(ms,2)
    %computes the mel-freequency cepstrum coefficients of each column of
    %the mel-frequency spectrum
    c(:, i) = dct(log(ms(:,i)));
end

%excludes the 0'th order cepstral coefficient; each column is the cepstrum
%coefficient for one of the blocked frames.
c = c(2:end , :);

end
