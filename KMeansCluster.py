"""
K-Means Clustering Algorithm

Partition n observations into K clusters

1. Initialization - K initial "means" centroids are generated at random
2. Assignment - K clusters are created by associating each observation w nearest centroid
3. Update - Centroid of clusters becomes new mean
4. Repeat until convergence
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})
print(df)

np.random.seed(200)
k = 3
centroids = {
	i+1: [np.random.randint(0,80), np.random.randint(0,80)]
	for i in range(k)
}
print("\n")
print("Initial Centroids")
print(centroids)
print("\n")

colormap = {1: 'r', 2: 'g', 3: 'b'}
fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], color='k')
for i in centroids.keys():
	plt.scatter(*centroids[i], color=colormap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()

# -----------------------------
# Initial Cluster assignment
# -----------------------------
def assignment(df, centroids):
	for i in centroids.keys():
		df['distance_from_{}'.format(i)] = (
			np.sqrt(
				(df['x'] - centroids[i][0]) ** 2
					+ (df['y'] - centroids[i][1]) ** 2
			)
		)
	centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
	df['closest'] = df.loc[:,centroid_distance_cols].idxmin(axis=1)
	df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
	df['color'] = df['closest'].map(lambda x: colormap[x])

	print("DF Closest Distance From Matrix:")
	print(df)
	print("\n")

	print("Centroid Cols:")
	print(centroid_distance_cols)
	print("\n")

	return [centroid_distance_cols, df]

[centroid_distance_cols, df] = assignment(df, centroids)
print(df.head())
print("\n")

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.2, edgecolor='k')
for i in centroids.keys():
	plt.scatter(*centroids[i], color=colormap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

# --------------------------------------------
# Update the cluster with new centroid means
# --------------------------------------------
import copy

old_centroids = copy.deepcopy(centroids)
def update(k):
	for i in centroids.keys():
		# average all x and y coordinates for a given centroid based on current clusters	
		centroids[i][0] = np.mean(df[df['closest'] == i]['x']) 
		centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
	return k 

centroids = update(centroids)
print("New Centroids:")
print(centroids)
print("\n")

fig = plt.figure(figsize=(5,5))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.2, edgecolor='k')
for i in centroids.keys():
	plt.scatter(*centroids[i], color=colormap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

# Get all distances for each closest key and average 
'''
total_dists = {}
for x in centroid_distance_cols: 
	key = int(x.lstrip('distance_from_'))
	idx = df.index[df.loc[:, 'closest'] == key].tolist()
	print(idx)
	#dists = df[x]
	dists = df[x][idx]	
	total_dists[x] = np.sum(dists)	
	print(dists)
print("\n")
print("Total Dists:")
print(total_dists)
print("\n")
'''

# Repeat the assignment processes with the new centroids
[centroid_distance_cols, df] = assignment(df, centroids)

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.2, edgecolor='k')
for i in centroids.keys():
	plt.scatter(*centroids[i], color=colormap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

# Continue until all assigned categories do not change
while True:
	old_centroids = df['closest'].copy(deep=True)
	centroids = update(centroids)
	[centroid_distance_cols, df] = assignment(df, centroids)
	if old_centroids.equals(df['closest']):
		break

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.2, edgecolor='k')
for i in centroids.keys():
	plt.scatter(*centroids[i], color=colormap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()


