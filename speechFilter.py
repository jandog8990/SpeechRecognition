import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# Pre-emphasis
Fs, signal = scipy.io.wavfile.read('test/s1.wav')
#emph_signal = signal
pre_emph = 0.95
emph_signal = np.append(signal[0], signal[1:] - pre_emph*signal[:-1]) 
signal_len = len(emph_signal)

#s = signal[8224:8224+10]
#print("S0 = " + str(s[0]))
#print(s)
#print(s[1:])
#print(s[:-1])

frame_size = 0.025		# milliseconds
frame_stride = 0.01 	# stride for each frame
frame_len = 0.025*Fs	# ms to samples using Fs
frame_step = 0.01*Fs

# calculate sizes of frames
frame_len = int(round(frame_len))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step))

# pad signal to make sure each frame has equal number of samples 
pad_signal_len = num_frames * frame_step + frame_len
z = np.zeros(pad_signal_len - signal_len)
pad_signal = np.append(emph_signal, z)	
ar = np.arange(0, 10)
num_frames_step = num_frames * frame_step

# frames from the index and shifted index matrices
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
#N = len(n);
#w = np.zeros(wlen)
w = 0.54 - 0.46*np.cos((2*np.pi*n)/(N-1))

# Apply Hamming window to data frames
frames *= w
#NFFT = 512

# Compute the FFT and PSD of the spectrum
fft_frames = np.fft.fft(frames)
fft_frames = fft_frames[:, 0:int(round(N/2))]
psd_frames = 1/(Fs*N)*(np.abs(fft_frames)**2)
psd_frames[1:len(psd_frames)-1] = 2*psd_frames[1:len(psd_frames)-1]
#psd_frames = 2*psd_frames

# Create numpy array of frequencies and half the PSD to remove periodicity?
# QUESTION: How to plot the frequencies of the current block
freq = np.arange(0, Fs/2, Fs/N)
row_sample = int(round(len(fft_frames)*2/4))	# sample at the half point of the matrix 
print("Row sample = " + str(row_sample))

#row_sample2 = int(round(len(fft_frames)*1/4))	# sample at the quarter point of the matrix 
sample_frames = frames[row_sample]			# row selection for FFT frames 
#sample_frames2 = frames[row_sample2]		# row selection for FFT frames 
mirror_psd = psd_frames[row_sample]	# row selection for FFT frames 
#mirror_psd = mirror_psd[0:int(round(len(mirror_psd)/2))]

print("\n")
print("Frame Block Indices:")
print(iMatrix.astype(np.int32, copy=False))
print("\n")

print("Hamming Window Frames (size = " + str(frames.shape) + "):")
print(frames)
print("\n")


print("Original signal len = " + str(signal_len))
print("Pad signal len = " + str(pad_signal_len))
print("N = frame_len = " + str(frame_len))
print("FS/2 = " + str(Fs/2))
print("Fs/N = " + str(Fs/N))
print("\n")

print("Frames size = " + str(frames.shape))
print("FFT Frames size = " + str(fft_frames.shape))
print("PSD Frames size = " + str(psd_frames.shape))

print("\n")
print("Frequency size = " + str(len(freq)))
print("Sample frames 1 size = " + str(len(sample_frames)))
#print("Sample frames 2 size = " + str(len(sample_frames2)))
print("Mirror PSD Frames size = " + str(mirror_psd.shape))
print("\n")

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

plt.figure
plt.imshow(10*np.log10(psd_frames.T), cmap=plt.cm.jet, aspect='auto');
ax = plt.gca()
ax.invert_yaxis()
plt.title("PSD Short Fourier Transform")
plt.ylabel("Frequency (FFT Number)")
plt.xlabel("Time (frame number)")
plt.show()
