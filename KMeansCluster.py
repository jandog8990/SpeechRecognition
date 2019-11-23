"""
K-Means Clustering Algorithm

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
#%matplotlib inline
    
# Columns of 2D vectors (i.e. x, y are vectors in space)
df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})
print("Data Frame:")
print(df)
print("\n")

# ----------------------------------------------------------------------
# Initial Cluster assignment (partitions x, y coords to each cluster)
# ----------------------------------------------------------------------
def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                    + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    
    # This finds the min for each row and assigns it to Distance Column
    df['closest'] = df.loc[:,centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colormap[x])

    return [centroid_distance_cols, df]

# --------------------------------------------
# Update the cluster with new centroid means
# --------------------------------------------
def update(k):

    for i in centroids.keys():
        # average all x and y coordinates for a given centroid based on current clusters
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        
    return centroids 

# Initialize the Centroids X, Y objects
np.random.seed(200)
k = 3 
centroids = {
    i+1: [np.random.randint(0,80), np.random.randint(0,80)]
    for i in range(k)
}
print("\n")
print("Initial Centroids")
print("Centroid type = " + str(type(centroids)))
print("Centroid len = " + str(len(centroids)))
print(centroids)
print("\n")

# Scatter the initial X, Y parameters
colormap = {1: 'r', 2: 'g', 3: 'b'}
fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], color='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colormap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()

# Initial Assign the X, Y points to the new centroids
print("Initial Assignment Dataframe to the Centroids:")
[centroid_distance_cols, df] = assignment(df, centroids)

print("DF Closest Distance From Matrix:")
print(df)
print("\n")

print("Centroid Cols:")
print(centroid_distance_cols)
print("\n")

# Plot the current centroids with all X, Y points
fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.2, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colormap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

# Continue until all assigned categories do not change
print("While Loop:")
count = 0
while True:
    print("COUNT = " + str(count))
    old_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    [centroid_distance_cols, df] = assignment(df, centroids)
    
    print("Updated Centroids:")
    print(centroids);
    print("\n");
    
    print("Assigned Data Frame:")
    [M, N] = df.shape
    print(df)
    print("\n")
    
    # Get all distances for each closest key and average 
    total_dists = {}
    total_sum = 0
    print("Total Distortion:");
    for x in centroid_distance_cols: 
        print("Distance col = " + str(x));    
        key = int(x.lstrip('distance_from_'))
        idx = df.index[df.loc[:, 'closest'] == key].tolist()
        print("index = " + str(idx))
        dists = df[x][idx]    
        total_dists[x] = np.sum(dists)
        total_sum = total_sum + total_dists[x]
        print(dists)
        print("\n")
    aver_dist = total_sum/M
    print("Total Dists:")
    print(total_dists)
    print("    => Total dist. = " + str(total_sum))
    print("    => Aver. Distortion = " + str(aver_dist))
    print("\n")

    fig = plt.figure(figsize=(5,5))
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.2, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colormap[i])
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()
    
        # If the old centroids equal the new we have convergence
    if old_centroids.equals(df['closest']):
        break
    count = count + 1


