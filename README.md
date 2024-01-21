# SpeechRecognition
This is a new repository for Math 572 Harmonic Analysis - Speech Recognition

This project consists of two main sections (Signal Processing and LGB/K-means Clustering)

### How to run code
$ python MFCC.py - takes input audio files and converts to frequency spectrum
$ python

**Signal Processing**
- Short Time FFT (ie windowing) for preserving frequency changes over time
- Conversion to Cepstrum domain for better understanding of speech freqs
- MFCC (Mel Frequency Cepstral Coefficients) non-linear representation of signal

**LGB & K-means Clustering**
- Main goal is to create cluster for codebooks containing codewords (ie clusters)
- Nearest neighbor search for clustering given speaker into speech vector
- LGB used to reset the centroids in K-means clustering

**Testing**
- Training will be done on multiple speech signals
- Testing will be done on unknown speech signals from audience
