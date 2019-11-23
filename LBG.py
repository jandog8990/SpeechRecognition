"""
LBG Clustering Algorithm

Partition n observations into K clusters

TODO: This will be transferred to an LBG algorithm to calculate
        acoustic vectors for speech recognition

1. Initialization - K initial "means" centroids are generated at random
2. Assignment - K clusters are created by associating each observation w nearest centroid
3. Update - Centroid of clusters becomes new mean
4. Repeat until convergence
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import string

# Load the training data from the MFCC Matrices from ProcessSpeaker file
# TODO THis is a test to see if we can even get the codes
# Load the current MFCC Lifted numpy arrays from file
npy_dir = "./train/";
N = 6;
delta = 0.01;   # delta for splitting the centroids
k = 16;         # number of codewords per codebook
MFCC = {}
# From the MFCC we will use LBG algorithm to find distances (clustering)
# NOTE We will have N Columns of M dimensional vectors (i.e. 19-D vectors)
for i in np.arange(1, N):
    filename = npy_dir + "s" + str(i) + "_mfcc_lift.npy"
    mfcc = np.load(filename)
    MFCC[i] = mfcc

# Will test on one speaker MFCC matrix
mfcc = MFCC[1]
[M, N] = mfcc.shape
test_mat = mfcc[:, 0:2]
print("mfcc:");
print(mfcc.shape);
print(mfcc);
print("\n");

# Pull the first row and average for a test
row1 = mfcc[1,:]
row1m = np.mean(row1)
row1s = np.sum(row1)
print("Row 1:")
print(row1.shape)
print(row1)
print("Row 1 sum = " + str(row1s))
print("Row 1 mean NP = " + str(row1m))
print("Row 1 mean = " + str(float(row1s/N)))
print("\n")

# Create the initial centroid
centroids = {}
centroid = np.mean(mfcc, axis=1)
print("Centroid:")
print(centroid.shape)
print(centroid)
print("\n");

# Create the split using the delta given perturbation factor
centroids[0] = centroid - delta
centroids[1] = centroid + delta
print("Centroid Dict:")
print(centroids.shape)
print(centroids)
print("\n")

        