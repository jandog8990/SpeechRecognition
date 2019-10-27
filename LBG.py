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

class LBG():

	# Initialize vars for the class
	eps = 0		# epsilong threshold for splitting
	K = 0		# init number of clusters
	D = None	# distortion vectors
	Y = None	# clustering vectors (reconstruction values)

	# Constructor
	def __init__(self, eps, K):
		self.eps = eps
		self.K = K

	# Run LBG/K-Means clustering on input training vector matrix
	def run_clustering(self, training_vectors):
		eps = self.eps
		K = self.K
		[M,N] = training_vectors.shape	
		print("Run Clustering:")
		print("Training vector size = " + str(training_vectors.shape))
		print("Epsilon : K = " + str(eps) + " : " + str(K))

		# Pull 2D sample vectors from the training vector matrix (for 2D plots)
		start = int(np.floor(M/2))
		end = start + 1
		print("M = " + str(M))	
		s1 = training_vectors[start,:]
		s2 = training_vectors[end,:]
		print("S1 sample size = " + str(len(s1)))
		print("S2 sample size = " + str(len(s2)))
		print("\n")
		
		plt.figure()
		plt.scatter(s1, s2, facecolors='none', edgecolors='b')
		plt.show()

		return 0

