function test(testdir, n, code)
% Speaker Recognition: Testing Stage
%
% Input:
%       testdir : string name of directory contains all test sound files
%       n       : number of test files in testdir
%       code    : codebooks of all trained speakers
%
% Note:
%       Sound files in testdir is supposed to be: 
%               s1.wav, s2.wav, ..., sn.wav
%
% Example:
%       >> test('C:\data\test\', 8, code);

for k = 1:n                     % read test sound file of each speaker
%     file = sprintf('%s/s%d.wav', testdir, k);
%     [s, fs] = audioread(file);   
%         
%     v = mfcc(s, fs);            % Compute MFCC's

    % Pull MFCC from Python mat files
    file = testdir + string(k) + "_mfcc.mat";
    mfcc = load(file);
    v = mfcc.mfcc;
    MFCC{k} = v;
   
    distmin = inf;
    k1 = 0;
   
    for l = 1:length(code)      % each trained codebook, compute distortion
        d = disteu(v, code{l}); 
        dist = sum(min(d,[],2)) / size(d,1);
      
        if dist < distmin
            distmin = dist;
            k1 = l;
        end      
    end
   
    msg = sprintf('Test Speaker %d matches with Train Speaker %d', k, k1);
    disp(msg);
end