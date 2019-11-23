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
N = 6;          # number of speakers (used for npy output)
MCOUNT = 1;     # centroid tracking index (used for split)
K = 16;         # number of codewords per codebook
delta = 0.01;   # delta for splitting the centroids
eps = 4.9;      # epsilon distortion limit for centroids

# Color map for centroids and data partition clusters
colormap = {1: 'blueviolet', 2: 'forestgreen', 3: 'deeppink', 4: 'dodgerblue', 5: 'indigo', 6: 'gold', 7: 'darkgray', 8: 'red', 9: 'lawngreen', 10: 'sienna', 11: 'olive', 12: 'salmon', 13: 'steelblue', 14: 'mediumblue', 15: 'purple', 16: 'peru'}

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

# --------------------------------------------------------------------------------
# Centroid split when the distortion drops below threshold (epsilon) 
# --------------------------------------------------------------------------------
def centroid_split(M, centroids):
    print("Centroid Split:")
    print("M = " + str(M))
    print("delta = " + str(delta)) 
    print("centroids keys:");
    print(centroids.keys())
    print("\n")
   
    # Update centroids with close centroids
    centroids[2*M - 1] = centroids[M] + delta
    centroids[2*M] = centroids[M] - delta
    M = 2*M 

    return [M, centroids]

# --------------------------------------------------------------------------------
# Nearest-Neighbor Assignment (i.e. assign 19-d vectors to nearest centroids)
# --------------------------------------------------------------------------------
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
   
    # Set the minimum distances for each data point to nearest centroid
    df['closest'] = df.loc[:,centroid_distance_cols].idxmin(axis=1) # find the minimum of all cols
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colormap[x])

    return [centroid_distance_cols, df]

# --------------------------------------------------------------------------------
# Update the centroids according to the new averages of partitions
# --------------------------------------------------------------------------------


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
centroids[1] = np.mean(mfcc, axis=1)
print("Initial Centroid:")
print(len(centroids))
print(centroids)
print("\n");

# Create the split using the delta given perturbation factor
[MCOUNT, centroids] = centroid_split(MCOUNT, centroids)
print("Updated Centroids:")
print("MCOUNT = " + str(MCOUNT))
print(len(centroids))
print(centroids)
print("\n");


# Print the initial data along with the first centroid
# We will plot the first two columns
fig = plt.figure(figsize=(5,5))
plt.scatter(data_frame['g'], data_frame['h'], color='lightgray')
for i in centroids.keys():
    print("[centroid, color] = " + str(i) + ", " + str(colormap[i]) + "]") 
    print(*centroids[i][7:9]) 
    print("-----------------------"); 
    plt.scatter(*centroids[i][7:9], color=colormap[i])
plt.show()

# Calculate the initial neighbor partitions
[centroid_distance_cols, df] = nearest_neighbor(data_frame, centroids)
   
# Show the final DataFrame of minimum distances
print("Centroid Distance Cols:")
print(centroid_distance_cols)
print("\n")

print("Final Data Frame:")
print(df);
print("\n")
  
# Plot the updated centroids prior to the loop of fitting
fig = plt.figure(figsize=(5,5))
plt.scatter(data_frame['g'], data_frame['h'], color=df['color'], alpha=0.2, edgecolor='k')
for i in centroids.keys():
    print("[centroid, color] = " + str(i) + ", " + str(colormap[i]) + "]") 
    print(*centroids[i][7:9]) 
    print("-----------------------"); 
    plt.scatter(*centroids[i][7:9], color=colormap[i])
plt.show()
 
"""
Framework for Centroid Searching:
    1. Centroid Split (Step 2)
        -> Divide current centroids into two (2*MCOUNT) 
        -> Set the difference as delta separation
    2. Distortion check (Step 6)
        if (D[n-1] - D[n])/D[n] > epsilon:
            -> Step 3 (Nearest-neighbor search)
        else:
            -> Step 7 (Codebook check) 
    3 . Codebook check
        if (MCOUNT == K):
            -> MCOUNT = K, number of codewords for codebook is reached (BREAK)
        else:
            -> Step 2 (split the centroids) 
"""

