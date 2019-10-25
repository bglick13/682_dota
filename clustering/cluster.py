import os, sys, argparse, json
sys.path.append(os.path.abspath('..'))

import pickle
import pandas as pd
import numpy as np
import torch, umap

from IPython import embed

from sklearn import metrics
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from matplotlib import cm, offsetbox
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, CSS4_COLORS


def generate_clustering(embeddings, nodes, algorithm, reduction):
	embeddings = pickle.load( open(embeddings, mode = 'rb'), encoding = 'bytes')
	nodes = pickle.load( open(nodes, mode = 'rb'), encoding = 'bytes')
	hero_ids = pd.DataFrame(data = json.load( open( os.path.join('..', 'const', 'hero_ids.json'), mode='rb')))

	X = embeddings.sum(axis = 2)

	clustering = None
	if algorithm == 'kmeans':
		clustering = KMeans(n_clusters = 5, random_state=0).fit(X)
	elif algorithm == 'dbscan':
		tmp = (X - np.max(X,0))/(np.max(X,0) - np.min(X,0))
		clustering = DBSCAN(eps=.125, min_samples=10).fit(tmp)
	elif algorithm == 'spectral':
		clustering = SpectralClustering(n_clusters = 5, assign_labels='discretize', random_state=0).fit(X)

	y = clustering.labels_

	if reduction == 'tsne':
		X_tsne = TSNE(n_components=3, random_state=0).fit_transform(X)
		x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
		X_tight = (X_tsne - x_min) / (x_max - x_min)

		colors = np.random.choice(list(CSS4_COLORS.keys()), len(np.unique(y)))

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i in range(len(colors)):
			tmp = X_tight[y == i]
			ax.scatter(tmp[:,:1], tmp[:,1:2], tmp[:,2:3], c = colors[i])

		plt.xticks([]), plt.yticks([])
		plt.show()

	cluster_samples = []

	for cluster in np.unique(y):
		indices = np.arange(y.shape[0])
		if len(indices) < 5:
			cluster_samples.append(nodes[indices])
		else:
			random_sampling = np.random.choice(indices[y == cluster], 5)
			cluster_samples.append(nodes[random_sampling])

	to_return = []
	for cluster in cluster_samples:
		comp = []
		for sample in cluster:
			heros = []
			for hero in sample:
				heros.append(hero_ids[hero_ids['id'] == hero]['name'].item())
			comp.append(heros)
		to_return.append(comp)

	print(to_return)

	pickle.dump(np.array(to_return), open('hero_clusters.pkl', mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--embeddings', '-e', required = True, default= '../data/clustering/10k_embed.pickle')
	parser.add_argument(
		'--nodes', '-n', required = True, default = '../data/clustering/10k_nodes.pickle')
	parser.add_argument(
		'--algorithm', '-a', required = False, default = 'kmeans')
	parser.add_argument(
		'--reduction', '-r', required = False, default = 'tsne')
	args = parser.parse_args()

	generate_clustering(args.embeddings, args.nodes, args.algorithm, args.reduction)