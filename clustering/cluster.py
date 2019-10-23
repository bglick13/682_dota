import os, sys, argparse
sys.path.append(os.path.abspath('..'))

import pickle
import numpy as np
import torch, umap

from IPython import embed

from sklearn import metrics
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from matplotlib import cm, offsetbox
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def generate_clustering(file, algorithm, reduction):
	embeddings = pickle.load( open(file, mode = 'rb'), encoding = 'bytes')

	X = embeddings.sum(axis = 2)

	clustering = None
	if algorithm == 'kmeans':
		clustering = KMeans(n_clusters = 5, random_state=0).fit(X)
	elif algorithm == 'dbscan':
		clustering = DBSCAN(eps=3, min_samples=10)
	elif algorithm == 'spectral':
		clustering = SpectralClustering(n_clusters = 5, assign_labels='discretize', random_state=0).fit(X)

	y = clustering.labels_

	X_tsne = TSNE(n_components=3, random_state=0).fit_transform(X)
	x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
	X_tight = (X_tsne - x_min) / (x_max - x_min)

	plt.figure()
	ax = plt.subplot(111)
	for i in range(X.shape[0]):
		plt.text(X_tight[i, 0], X_tight[i, 1], str(y[i]),
				 color = plt.cm.Set1(y[i]/10.),
				 fontdict={'weight': 'bold', 'size': 9})

	if hasattr(offsetbox, 'AnnotationBbox'):
		shown_images = np.array([[1., 1.]])
		for i in range(X.shape[0]):
			dist = np.sum((X_tight[i] - shown_images) ** 2, 1)
			if np.min(dist) < 4e-3:
				continue
			shown_images = np.r_[shown_images, [X_tight[i]]]

	plt.xticks([]), plt.yticks([])
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--embedding', '-e', required = True)
	parser.add_argument(
		'--algorithm', '-a', required = False, default = 'kmeans')
	parser.add_argument(
		'--reduction', '-r', required = False, default = 'TSNE')
	args = parser.parse_args()

	generate_clustering(args.embedding, args.algorithm, args.reduction)