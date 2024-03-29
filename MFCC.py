#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mel Frequency Cepstrum Coefficients class

This moves low and high frequencies to different sides 
of the spectrum thus giving us a non-linear output 
for human auditory perception

@author: alejandrogonzales
"""
import numpy as np
from scipy.fftpack import dct

# Mel Frequency Cepstrum Coefficients Class
# TODO remove all refs to self and replace with local vars
class MFCC():

	# Initialize input constructor vars
	Afcc = 1125	# amplitude of MFCC calculation (default see Mel scale equation)	
	#Afcc = 2595 # amplitude of MFCC calculation (default see Mel scale equation)	
	Bfcc = 700	# amplitude for exponential calculation (see Mel scale equation)	
	Nfft = 0	# number of FFT pts	
	Nfcc = 0	# number of MFCC filterbanks	
	Fs = 0		# sampling frequency	
	lower_freq = 0
	upper_freq = 0
	mel_pts = None	# init mel scale array
	freq_pts = None	# init frequency array
	bin_pts = None	# frequency bin array

	# Constructor
	def __init__(self, Nfft, Nfcc, Fs, lower_freq, upper_freq):
		self.Nfft = Nfft	
		#self.Nfcc = Nfcc+2	# add two to account for indep. endpoints	
		self.Nfcc = Nfcc	
		self.Fs = Fs	
		print("Nfft = " + str(Nfft))
		print("Nffc = " + str(Nfcc))
		print("Fs = " + str(Fs))
		print("\n")
		self.lower_freq = lower_freq
		self.upper_freq = upper_freq 

		# Initialize the mel frequency scale
		mel_pts = np.zeros(Nfcc+2)
		freq_pts = np.zeros(Nfcc+2)

	# Fit the mel frequency coefficients
	def fit(self):
		# TODO Consider create a function to calculate all mel coeffs

		# Fit the mel scale, frequency pts and frequency bin pts (filterbank frames) 
		mfccMap = {}	
		mel_lower = self.Afcc*np.log(1 + self.lower_freq/self.Bfcc)
		mel_upper = self.Afcc*np.log(1 + self.upper_freq/self.Bfcc)
        #mel_lower = self.Afcc*np.log10(1 + self.lower_freq/self.Bfcc)
		#mel_upper = self.Afcc*np.log10(1 + self.upper_freq/self.Bfcc)
		self.mel_pts = np.linspace(mel_lower, mel_upper, self.Nfcc+2)	
		self.freq_pts = self.Bfcc*(np.exp(self.mel_pts/self.Afcc) - 1) 
		#self.freq_pts = self.Bfcc*(10**(self.mel_pts/self.Afcc) - 1) 
		self.bin_pts = np.floor((self.Nfft)*self.freq_pts/self.Fs)

		mfccMap['mel'] = self.mel_pts
		mfccMap['freq'] = self.freq_pts
		mfccMap['bin'] = self.bin_pts	
	
		return mfccMap

	# Calculate the filterbanks (i.e. triangular filters)
	def calc_filter_banks(self):
		# initalize filterbank matrix and other dep. vars 
		Nfcc = self.Nfcc
		bin_pts = self.bin_pts

		# Matrix of filterbanks (rowlen = num_filteranks, colen = numfft)
		fbank = np.zeros((Nfcc, int(np.floor((self.Nfft/2+1)))))
		# Loop through the rows of the filterbank matrix (one row per bank)	
		for m in np.arange(1, Nfcc+1):
			f_m_minus = int(bin_pts[m-1])	# left triangle pt
			f_m = int(bin_pts[m])			# center triangle pt
			f_m_plus = int(bin_pts[m+1])	# right triangle pt

			# Loop through left triangular pts (see Hm(k) function)	
			for k in np.arange(f_m_minus, f_m):
				fbank[m-1, k] = (k - bin_pts[m-1])/(bin_pts[m] - bin_pts[m-1])	
			
			# Loop through right triangular pts (see Hm(k) function)
			for k in np.arange(f_m, f_m_plus):
				fbank[m-1, k] = (bin_pts[m+1] - k)/(bin_pts[m+1] - bin_pts[m])
				

		return fbank	

    # Calculate the Mel Frequency Cepstrum Coefficients (ie acoustic vector)
    # Each utterance is transformed into sequence of acoustic vectors
	def calc_mfcc_dct(self, filter_banks, num_mfcc):
		mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
		mfcc = mfcc[:, 1:(num_mfcc+1)]	# pull the 2nd to the num_mfcc+1 entries	
		
		return mfcc
	
	# Apply sin lifter to the MFCC matrix
	def calc_mfcc_lift(self, mfcc, num_lift):
		# apply liftering to remove high quefrencies from the MFCCs
		[nframes, ncoeffs] = mfcc.shape	# get the coefficients
		n = np.arange(ncoeffs)			# indices for mfcc coefficients
		slift = 1 + (num_lift/2)*np.sin((np.pi*n)/num_lift)	# sin lifter for high quefrencies
		mfcc_lift = mfcc*slift	


		# TODO To improve the MFCCs, that is to reduce the dependency on higher and
		# and lower frequencies in the vocal tract we need to apply liftering, that
		# is filtering in the quefrency domain (ie cepstrum domain)

		return mfcc_lift
