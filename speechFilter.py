import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# Pre-emphasis
Fs, signal = scipy.io.wavfile.read('test/s1.wav')
pre_emph = 0.97
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
#frames *= w

print("\n")
print("Frame len = " + str(frame_len))
print("Frame step = " + str(frame_step))
print("Num frames = " + str(num_frames)) 
print("\n")
print("Signal len = " + str(signal_len))
print("Pad signal len = " + str(pad_signal_len))
print("Len zeros = " + str(len(z)))
print("Num frames * frame step = " + str(num_frames_step))
print("\n")
print("iMatrix1 size = " + str(iMatrix1.shape))
print("iMatrix2 size = " + str(iMatrix2.shape))

print("Index Matrix:")
print(iMatrix1)
print("\n")
print("Shifted Matrix:")
print(iMatrix2)
print("\n")
print("Shifted Index Matrix:")
print(iMatrix)
print("\n")
print("Index Matrix (int32):")
print(iMatrix.astype(np.int32, copy=False))
print("\n")

print("Signal Frames:")
print(frames)
print("\n")

plt.plot(w)
plt.show()
