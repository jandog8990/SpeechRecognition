"""
LBG Algorithm (Linde Buzo and Gray Clustering algorithm)

Similar to K means by computing distortions (ie distances btwn
training vectors and centroids) and determing which vector is
closest with which centroid.

Goal is to minimize total distortion for training set.
K-means ithout an initial cluster K and also without 
knowing the number of clusters beforehand

@author: alejandrogonzales
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib inline

"""
TODO: This function currently uses two 20-D vectors for training
the goal is to use all N frames (i.e. 98 frames) consisting of
20-samples each (i.e. each sample represents an MFCC coefficient)
"""
class LBG():

    # Initialize vars for the class
    eps = 0        # epsilong threshold for splitting
    K = 0        # init number of clusters
    D = None    # distortion vectors
    Y = None    # clustering vectors (reconstruction values)

    # Constructor
    def __init__(self, eps, K):
        self.eps = eps
        self.K = K

    # Initial Cluster assignment using centroid vectors
    def assignment(self, df, centroids, colormap):
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

    # Generate test centroids
    ''' 
    def test_centroids(self, df, colormap):

        return centroids
    '''

    # Run LBG/K-Means clustering on input training vector matrix
    def run_clustering(self, training_vectors):
        eps = self.eps
        K = self.K
        K = 2    
        [M,N] = training_vectors.shape
        print("Training Vectors Shape:")
        print(training_vectors.shape)
        print("\n")
        
        # Compute the mean of all rows (that is the Mth coeff for N frames)    
        centroid = training_vectors.mean(1)    # 1st dim is rows

        #centroid = centroid.T    
        clen = len(centroid)
        csplit = K
        centroid_matrix = np.tile(centroid, (csplit,1))
        centroid_matrix = centroid_matrix.T
        centroid_matrix[:,0] = centroid_matrix[:,0] - eps
        centroid_matrix[:,1] = centroid_matrix[:,1] + eps

        print("Run Clustering:")
        print("Training vector size = " + str(training_vectors.shape))
        print("Epsilon : K = " + str(eps) + " : " + str(K)) 
        print("Centroid Matrix Shape:")
        print(centroid_matrix.shape)
        print("\n")
        # Colors for centroid and training vectors
        #colors = np.random.rand(M)
        #color = colors[0]

        self.run_lbg(K, training_vectors, centroid_matrix)        

        return 0

    # Run the LBG algorithm on the current training vectors
    def run_lbg(self, k, training_vectors, centroid_matrix):
        print("RUN LBG:")
        print(training_vectors.shape)
        print("\n")
        [M,N] = training_vectors.shape
        start = int(np.floor(M/2))    
        end = start + 1
        
        # Sample 2D vectors for training    
        xvec = training_vectors[start,:]
        yvec = training_vectors[end,:]

        # Create the centroid
        [Mc, Nc] = centroid_matrix.shape
        print("Centroid Matrix:")
        print(centroid_matrix.shape)
        print(centroid_matrix)
        print("\n")

        """
        plt.figure()
        scat = plt.scatter(xvec, yvec, facecolors='none', edgecolors='b')
        plt.show()
        """

        """
        # Test XY data
            'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
            'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
        """
        # Create DataFrame for the 2D train vectors    
        df = pd.DataFrame({
            'x': xvec,
            'y': yvec
        })
        print(df)
        print("\n")
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

        '''
        fig = plt.figure(figsize=(5,5))
        plt.scatter(df['x'], df['y'], facecolors='none', color='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colormap[i])
        plt.xlim(0,80)
        plt.ylim(0,80)
        plt.show()
        '''

        # WTF Am i doing here and why is it random?
        #np.random.seed(200)
            #i+1: [np.random.randint(0,Mc), np.random.randint(1,Nc)]
        ran_row = np.random.randint(0, Mc) 
        centroid_samp1 = centroid_matrix[start,:]
        centroid_samp2 = centroid_matrix[end,:]
        centroids = []
        
        # Plot the sample points for the centroid
        '''
        fig = plt.figure(figsize=(5,5))
        plt.scatter(df['x'], df['y'], facecolors='none', color='b')
        plt.scatter(cx, cy, c='r')
        plt.show()
        '''
        
        # Colormap for centroids
        colormap = {1: 'r', 2: 'g'}
#        centroids = self.test_centroids(df, colormap)
#        centroids = 

        [centroid_distance_cols, df] = self.assignment(df, centroids, colormap)
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
        """
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
        """
