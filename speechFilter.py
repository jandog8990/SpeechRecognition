import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# Local import classes
from MFCC import MFCC 

# Pre-emphasis
Fs, signal = scipy.io.wavfile.read('test/s1.wav')
pre_emph = 0.95
emph_signal = np.append(signal[0], signal[1:] - pre_emph*signal[:-1]) 
signal_len = len(emph_signal)

# Frame sizing and steps (ie overlap stride)
frame_size = 0.025		# milliseconds
frame_stride = 0.01 	# stride for each frame
frame_len = 0.025*Fs	# ms to samples using Fs
frame_step = 0.01*Fs

# Calculate integer sizes of frames
frame_len = int(round(frame_len))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step))

# Pad signal to make sure each frame has equal number of samples 
pad_signal_len = num_frames * frame_step + frame_len
z = np.zeros(pad_signal_len - signal_len)
pad_signal = np.append(emph_signal, z)	
ar = np.arange(0, 10)
num_frames_step = num_frames * frame_step

# Frames from the index and shifted index matrices
iMatrix1 = np.tile(np.arange(0, frame_len), (num_frames, 1))
iMatrix2 = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1))
iMatrix2 = iMatrix2.T  
iMatrix = iMatrix1 + iMatrix2
frames = pad_signal[iMatrix] 

# Apply Hamming Window to individual frames FFT assumes infinite data
# windowing will reduce spectral leakage
N = frame_len 
n = np.arange(0, N, 1)
wlen = len(n)
w = 0.54 - 0.46*np.cos((2*np.pi*n)/(N-1))

# Apply Hamming window to data frames
frames *= w

# Compute the FFT and PSD of the spectrum
fft_frames = np.fft.fft(frames)
[mfft, nfft] = fft_frames.shape
Nfft = nfft
psd_frames = 1/(Fs*Nfft)*(np.abs(fft_frames)**2)

# Split the FFT frames and PSD frames in half
psd_frames = psd_frames[:, 0:int(round(Nfft/2+1))]
fft_frames = fft_frames[:, 0:int(round(Nfft/2+1))]
psd_frames = psd_frames.T

print("Fs = " + str(Fs))
print("Fs/2 = " + str(Fs/2))
print("Fs/N = " + str(Fs/N))
print("\n")

print("Frames size = " + str(frames.shape))
print("FFT Frames size = " + str(fft_frames.shape))
print("PSD Frames size = " + str(psd_frames.shape))
print("\n")

# -------------------------------------------------------------------
# Mel Frequency Cepstrum Coeffs - Calculations
# TODO: Normalize the PSD and MFCC Spectrums only when plotting
#
# 1. 26 Triangular filters
# 2. Multiply power spectrum by each filt
# 3. Add up coefficients (26 give us energy of filterbank)
# 4. Take log of each of 26 energies (26 log filterbank energies)
# 5. DCT of 26 log filterbank energies (26 cepstral coeffs)
# 6. First 12-13 of 26 are kept
# 7. Resulting features (12 numbers per frame) are MFCCs
# -------------------------------------------------------------------
lower_freq = 0		# lower frequency for mel calc
upper_freq = Fs/2	# upper frequency for mel calc	
Nfcc = 30	# number of MFCC filterbanks (i.e. triangle filters) 

# Initialize the MFCC mel scale, freqs and bins
freq = np.arange(0, Fs/2, Fs/(Nfft+1))
mfcc = MFCC(Nfft, Nfcc, Fs, lower_freq, upper_freq)
mfcc_out = mfcc.fit() 
mel_pts = mfcc_out['mel']
freq_pts = mfcc_out['freq']
bin_pts = mfcc_out['bin']

# Create the filterbanks and normalize from the mean
filterbanks = mfcc.calc_filter_banks()

[fm, fn] = filterbanks.shape
#freqs = np.tile(freq, (fm,1))

# Compute the dot product between the power spectrum and filterbanks
mfcc_fbanks = np.dot(psd_frames.T, filterbanks.T)

# Normalize the PSD frames for better contour viewing
psd_frames = 20*np.log10(psd_frames)
#psd_frames -= (np.mean(psd_frames, axis=0) + 1e-8)

# Normalize the MFCC PSD Banks for better contours
mfcc_fbanks = 20*np.log10(mfcc_fbanks)
mfcc_fbanks -= (np.mean(mfcc_fbanks, axis=0) + 1e-8)

plt.figure(1)
for i in range(0,fm):
    plt.plot(filterbanks[i,:])
plt.title("MFCC Filterbank:")
plt.show()

plt.figure(2)
plt.imshow(psd_frames, cmap=plt.cm.jet, aspect='auto');
ax = plt.gca()
ax.invert_yaxis()
plt.title("PSD Short Fourier Spectrum")
plt.ylabel("Frequency (FFT Number)")
plt.xlabel("Time (frame number)")
plt.show()

plt.figure(3)
plt.imshow(mfcc_fbanks.T, cmap=plt.cm.jet, aspect='auto')
ax = plt.gca()
ax.invert_yaxis()
plt.title('PSD * MFCC Filterbank Spectrum')
plt.ylabel("Frequency (FFT Number)")
plt.xlabel("Time (frame number)")
plt.show()

## Plot sample frame with FFT PSD
#plt.figure
#plt.subplot(211)
#plt.plot(sample_frames)
#plt.title("PSD Short Fourier Transform")
#plt.subplot(212)
#plt.plot(10*np.log10(mirror_psd))
#plt.ylabel("Power/frequency (dB/Hz)")
#plt.xlabel("Frequency (Hz)")
#plt.show()
