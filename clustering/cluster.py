import argparse
import json
import os
import sys

sys.path.append(os.path.abspath('..'))

import pickle
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
from matplotlib.colors import CSS4_COLORS


def generate_class_clustering(nodes, algorithm, reduction):
	nodes = pickle.load( open(nodes, mode = 'rb'), encoding = 'bytes')
	hero_info = pd.DataFrame(data = json.load( open( os.path.join('..', 'const', 'hero_ids.json'), mode='rb')))

	classes = ["Carry", "Support", "Nuker", "Disabler", "Jungler", "Durable", "Escape", "Pusher", "Initiator"]
	
	roles = [[hero_info[hero_info['id'] == id]['roles'].item() for id in node] for node in nodes]
	# comps = [list(set().union(roles[i][0], roles[i][1], roles[i][2], roles[i][3], roles[i][4])) for i in range(len(roles))]
	# data = [[int(role in comps[i]) for role in classes] for i in range(len(comps))]
	comps = [(roles[i][0] + roles[i][1] + roles[i][2] + roles[i][3] + roles[i][4]) for i in range(len(roles))]
	data = np.array([[comps[i].count(role) for role in classes] for i in range(len(comps))])
	data = data / np.sum(data, axis = 1).reshape(-1,1)
	X = pd.DataFrame(data, columns = classes)
	kmeans = KMeans(n_clusters = 8, random_state = 0).fit(X)

	y = kmeans.labels_

	if reduction == 'tsne_2':
		X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
		x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
		X_tight = (X_tsne - x_min) / (x_max - x_min)

		colors = np.random.choice(list(CSS4_COLORS.keys()), len(np.unique(y)))

		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(len(colors)):
			tmp = X_tight[y == i]
			ax.scatter(tmp[:,:1], tmp[:,1:2], c = colors[i])

		plt.xticks([]), plt.yticks([])
		plt.show()

	if reduction == 'tsne_3':
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

	if reduction == 'tsne_2':
		X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
		x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
		X_tight = (X_tsne - x_min) / (x_max - x_min)

		colors = np.random.choice(list(CSS4_COLORS.keys()), len(np.unique(y)))

		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(len(colors)):
			tmp = X_tight[y == i]
			ax.scatter(tmp[:,:1], tmp[:,1:2], c = colors[i])

		plt.xticks([]), plt.yticks([])
		plt.show()

	if reduction == 'tsne_3':
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
		'--embeddings', '-e', required = False, default= '/data/clustering/10k_embed.pickle')
	parser.add_argument(
		'--nodes', '-n', required = False, default = '/data/clustering/10k_nodes.pickle')
	parser.add_argument(
		'--algorithm', '-a', required = False, default = 'kmeans')
	parser.add_argument(
		'--reduction', '-r', required = False, default = 'tsne_2')
	args = parser.parse_args()

	generate_class_clustering(args.nodes, args.algorithm, args.reduction)
	# generate_clustering(args.embeddings, args.nodes, args.algorithm, args.reduction)