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
eps = 4.9;      # epsilon distortion limit for centroids
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

# Create the DataFrame with all of the MFCC data vectors
data = {}
print("Data:")
for i in range(M):
    axis = string.ascii_lowercase[i]
    row = mfcc[i,:]
    data[axis] = row
data_frame = pd.DataFrame(data)
print("Data Frame:")
print(data_frame.shape)
print(data_frame)
print("\n")

# Nearest-Neighbor Assignment (i.e. assign 19-d vectors to nearest centroids)
def nearest_neighbor(df, centroids):
    # data frame keys
    df_keys = df.keys()

    print("Centroids Length:")
    print(len(centroids));
    print("\n")

    print("DataFrame keys:")
    print(df_keys)
    print("\n")

    for i in centroids.keys():
        diff_sum = 0 
        for j in range(len(df_keys)):
            df_key = df_keys[j]
            ci = centroids[i][j]
            print("[i, j] = [" + str(i) + ", " + str(j) + "]"); 
            print("DF key = " + str(df_key))
            print("Centroid val = " + str(ci))
            
            # Sum of all the dimensional vectors
            #diff_sum = diff_sum + (df[df_key] - ci)
            diff_sq = (df[df_key] - ci)**2
            diff_sum = diff_sum + diff_sq 
            print("Dataframe axis difference:")
            print(len(diff_sq)) 
            print(diff_sq)
            print("\n")
        
        # Assign the final distance calculations to the distance from column
        distance_from = np.sqrt(diff_sum) 
        df['distance_from_{}'.format(i)] = distance_from
        print("Distance from " + str(i) + ":")
        print(distance_from)
        print("\n")

    # Set the distance from centroid column in the data frame
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    print("Centroid Distance Cols:")
    print(centroid_distance_cols)
    print("\n")
   
    # Set the minimum distances for each data point to nearest centroid
    df['closest'] = df.loc[:,centroid_distance_cols].idxmin(axis=1) # find the minimum of all cols
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
   
    # Show the final DataFrame of minimum distances
    print("Final Data Frame:")
    print(df);
    print("\n")



# Pull the first row and average for a test
row1 = mfcc[1,:]
row1m = np.mean(row1)
row1s = np.sum(row1)
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
print(centroids)
print("\n")

# Print the initial data along with the first centroid
# We will plot the first two columns
'''
colormap = {0: 'r', 1: 'g', 2: 'b', 3: 'p'}
fig = plt.figure(figsize=(5,5))
plt.scatter(data_frame['h'], data_frame['i'], color='k')
for i in centroids.keys():
    print("key = " + str(i));
    plt.scatter(*centroids[i][0:2], color=colormap[i])
plt.show()
'''

# Calculate the initial neighbor partitions
nearest_neighbor(data_frame, centroids)
