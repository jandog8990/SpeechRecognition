"""
LBG Clustering Algorithm

Partition n observations into K clusters

TODO: Need to train this on multiple speakers (consider cmd line loop)
        Also need to be able to run this with test data

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
K = 8;         # number of codewords per codebook
delta = 0.01;   # delta for splitting the centroids
eps = 0.001;      # epsilon distortion limit for centroids

# Color map for centroids and data partition clusters
colormap = {1: 'blueviolet', 2: 'forestgreen', 3: 'deeppink', 4: 'dodgerblue', 5: 'indigo', 6: 'gold', 7: 'darkgray', 8: 'red', 9: 'lawngreen', 10: 'sienna', 11: 'olive', 12: 'salmon', 13: 'steelblue', 14: 'mediumblue', 15: 'purple', 16: 'peru'}

MFCC = {}
# From the MFCC we will use LBG algorithm to find distances (clustering)
# NOTE We will have N Columns of M dimensional vectors (i.e. 19-D vectors)
for i in np.arange(1, N):
    filename = npy_dir + "s" + str(i) + "_mfcc_lift.npy"
    mfcc = np.load(filename)
    MFCC[i] = mfcc

# TODO Will use a loop over all speakers and pull MFCC for each
# Will test on one speaker MFCC matrix
mfcc = MFCC[1]
[M, N] = mfcc.shape

# Create the DataFrame with all of the MFCC data vectors
data = {}
axes = {} 
print("Data:")
for i in range(M):
    axis = string.ascii_lowercase[i]
    axes[i] = axis 
    row = mfcc[i,:]
    data[axis] = row
data_frame = pd.DataFrame(data)
print("Axes:")
print(len(axes))
print(axes)
print("\n")

print("Data Frame:")
print(data_frame.shape)
print(data_frame)
print("\n")

# --------------------------------------------------------------------------------
# Calculate average distortion betwen centroids and data points
# --------------------------------------------------------------------------------
def average_distortion(df, centroid_distance_cols):
    total_dists = {}
    total_sum = 0
    for x in centroid_distance_cols: 
        key = int(x.lstrip('distance_from_'))
        idx = df.index[df.loc[:, 'closest'] == key].tolist()
        dists = df[x][idx]    
        total_dists[x] = np.sum(dists)
        total_sum = total_sum + total_dists[x]
    aver_dist = total_sum/M
    
    return [total_sum, aver_dist]

# --------------------------------------------------------------------------------
# Centroid split when the distortion drops below threshold (epsilon) 
# --------------------------------------------------------------------------------
def centroid_split(M, centroids):
    cc = copy.deepcopy(centroids)
    centroid_keys = cc.keys()
    print("Centroid Split:")
    print("Len centroids = " + str(len(centroids)))  
    print("M = " + str(M))
    print("delta = " + str(delta)) 
    print("centroids keys:");
    print(centroid_keys)
    print("\n")

    # Loop through centroids and split each into two
    for i in centroid_keys: 
        # update centroids with close centroids
        centroids[2*i - 1] = centroids[i] * (1+delta)
        centroids[2*i] = centroids[i] * (1-delta)
   
    M = len(centroids)

    return [M, centroids]

# --------------------------------------------------------------------------------
# Nearest-Neighbor Assignment (i.e. assign 19-d vectors to nearest centroids)
# --------------------------------------------------------------------------------
def nearest_neighbor(MCOUNT, count, df, centroids):
   
    if (count == 10):
        if (MCOUNT >= 8):
            print("Nearest Neighbor:")
            print("------------------------------------------------------------------")
            print("Centroids (len = " + str(len(centroids)) + "):")
            print(centroids)
            print("\n")

    # loop through the centroids to calculate distances
    for i in centroids.keys():
        diff_sum = 0 
        # loop through axes for coordinates of each dimension 
        for j in range(len(axes)):
            axis = axes[j] 
            ci = centroids[i][j]
            
            # Sum of all the dimensional vectors
            diff_sq = (df[axis] - ci) ** 2
            diff_sum = diff_sum + diff_sq 
            
            if (count == 10):
                if (MCOUNT >= 8):
                    if (j <= 2): 
                        print("------------------------------------------------------------")
                        print("axis = " + str(axis))
                        print("df[" + str(axis) + "]:")
                        print(df[axis])
                        print("\n")
                        print("centroid[" + str(i) + ", " + str(j) + "]:")
                        print(ci);
                        print("Diff squared = " + str(diff_sq))
                        print("Diff sum = " + str(diff_sum))
                        print("------------------------------------------------------------")
                        print("\n")
        
        # Assign the final distance calculations to the distance from column
        distance_from = np.sqrt(diff_sum) 
        if (count == 10):
            if (MCOUNT >= 8):
                print("COUNT = " + str(count));
                print("Distance from [" + str(i) + "]:")
                print(distance_from)
                print("------------------------------------------------------------------")
                print("\n")
            
        df['distance_from_{}'.format(i)] = distance_from

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
def update_centroids(MCOUNT, count, df, centroids):
    
    # loop through the centroids and calculate the means of the closest vectors
    for i in centroids.keys():
        # loop through the axes in order to calculate averages 
        for j in range(len(axes)):
            axis = axes[j]
    
            if (count >= 9):
                if (MCOUNT >= 8):
                    print("Centroid Key:")
                    print(str(i))
                    print("[j, Axis] = [" + str(j) + ", " + str(axis) + "]"); 
                    print("\n")

                    print("DF Closest:")
                    print(df['closest'])
                    print(centroids)
                    print("\n")

            # average all dimension coordinates for a given centroid (i.e. [a, b, c, ...]
            centroids[i][j] = np.mean(df[df['closest'] == i][axis])
    
    return centroids

# Create the initial centroid
centroids = {}
centroids[1] = np.mean(mfcc, axis=1)
print("Initial Centroid:")
print(len(centroids))
print(centroids)
print("\n");

# Create the split using the delta given perturbation factor
[MCOUNT, centroids] = centroid_split(MCOUNT, centroids)
print("Split Centroids:")
print("MCOUNT = " + str(MCOUNT))
print(len(centroids))
print(centroids)
print("\n");


# Print the initial data along with the first centroid
# We will plot the first two columns
fig = plt.figure(figsize=(5,5))
plt.scatter(data_frame['g'], data_frame['h'], color='lightgray')
for i in centroids.keys():
    plt.scatter(*centroids[i][6:8], color=colormap[i])
plt.show()

# Calculate the initial neighbor partitions
[centroid_distance_cols, data_frame] = nearest_neighbor(0, 0, data_frame, centroids)
  
# Compute initial average distortion
[total_dist, mean_dist] = average_distortion(data_frame, centroid_distance_cols)

# Update the centroids for the new distances
centroids = update_centroids(0, 0, data_frame, centroids)

print("Update Centroids Out:")
print(len(centroids))
print(centroids)
print("\n")

#print("Final Data Frame:")
#print(data_frame.shape)
#print(data_frame);
#print("\n")   

#print("Total Distortion = " + str(total_dist))
#print("Average Distortion = " + str(mean_dist))
#print("\n")

# Plot the updated centroids prior to the loop of fitting
fig = plt.figure(figsize=(5,5))
plt.scatter(data_frame['g'], data_frame['h'], color=data_frame['color'], alpha=0.2, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i][6:8], color=colormap[i])
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
count = 0;
while True:
    if count == 11:
        break;
        
    # Old centroid closest vector (used for final checks) 
    old_centroids = data_frame['closest'].copy(deep=True)

    # Old mean distortion (i.e. D[n-1])
    old_mean_dist = mean_dist

    # Update centroids and calculate nearest neighbors
    [centroid_distance_cols, data_frame] = nearest_neighbor(MCOUNT, count, data_frame, centroids)
    [M, N] = data_frame.shape
    
    # Check the Data Frames for distances
#    dist_from = data_frame.filter(regex="distance_from")
#    for k in dist_from.keys():
#        print("Distance From [" + str(k) + "]:")
#        print(dist_from[k])
#        print("\n")
  
    # Compute average distortion and threshold distorition
    [total_dist, mean_dist] = average_distortion(data_frame, centroid_distance_cols)
    thresh_dist = (old_mean_dist - mean_dist)/mean_dist
    thresh_dist = abs(thresh_dist)
#    print("    => Total Distortion      = " + str(total_dist))
#    print("    => Old Aver Distortion   = " + str(old_mean_dist)) 
#    print("    => Aver. Distortion      = " + str(mean_dist))
#    print("    => Thresh. Distortion    = " + str(thresh_dist))
#    print("    => Epsilon (thresh)      = " + str(eps))  
#    print("\n")

    # Update the centroids
    centroids = update_centroids(MCOUNT, count, data_frame, centroids)
    if (count >= 9):
        if (MCOUNT >= 8):
            print("Updated Centroids:")
            print(len(centroids))
            print(centroids)
            print("\n")

    # Plot the updated centroids prior to the loop of fitting
    # TODO Will plot the final clusters from the LBG output 
#    fig = plt.figure(figsize=(5,5))
#    plt.scatter(data_frame['j'], data_frame['k'], color=data_frame['color'], alpha=0.2, edgecolor='k')
#    for i in centroids.keys():
#        plt.scatter(*centroids[i][12:14], color=colormap[i], edgecolor='k')
#    plt.show()

    # TODO This isn't being hit for the threshold epsilon check
    # First check if new centroids equal previous centroids (i.e. convergence)
    if (old_centroids.equals(data_frame['closest'])) or (thresh_dist < eps): 
        
        # Split the centroids 
        if (MCOUNT != K): 
#            print("Split Centroids:")
#            print("MCOUNT = " + str(MCOUNT))
#            print("Centroids in:") 
#            print(len(centroids))
#            print("\n")
            
            [MCOUNT, centroids] = centroid_split(MCOUNT, centroids)

            print("---------- Centroid Split K = " 
                  + str(MCOUNT) + " ----------")
            print("Count = " + str(count)) 
            print("MCOUNT = " + str(MCOUNT))
            print("Centroids out:") 
            print(len(centroids))
            print(centroids)
            print("\n")
            
            continue 
        else:
            print("BREAK!!!!");
            print("Count = " + str(count)) 
            print("MCOUNT == K == " + str(MCOUNT));
            print("Final Centroids:")
            print(len(centroids))
            print(centroids)
            print("\n")
            
            break;
    
    count = count + 1 
