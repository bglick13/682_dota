import json
import os
import sys

sys.path.append(os.path.abspath('..'))

import pickle
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans


class KmeansCluster(object):
	def __init__(self, centroids, train_data=None):
		self.classes = ["Carry", "Support", "Nuker", "Disabler", "Jungler", "Durable", "Escape", "Pusher", "Initiator"]
		self.hero_info = pd.DataFrame(data = json.load( open( os.path.join('..', 'const', 'hero_ids.json'), mode='rb')))
		self.centroids=centroids
		self.cluster = None
		if train_data is not None:
			self.cluster = KMeans(centroids, random_state=0)
			self.cluster.fit(train_data)

	def process_raw_data(self, input_X):
		'''
		This function currently expects to take in a pickled list
		that is a list of teams where each team is a list of hero IDs.

		Returns a pandas datafram object with the correct encoding
		for clustering. Encoding is based on hero roles in each team. '''
		roles = [[self.hero_info[self.hero_info['id'] == id]['roles'].item() for id in node] for node in input_X]
		comps = [(roles[i][0] + roles[i][1] + roles[i][2] + roles[i][3] + roles[i][4]) for i in range(len(roles))]
		data = np.array([[comps[i].count(role) for role in self.classes] for i in range(len(comps))])
		data = data / np.sum(data, axis = 1).reshape(-1,1)
		X = pd.DataFrame(data, columns = self.classes)

		return X

	def fit(self, X, centroids=None):
		X = self.process_raw_data(X)

		cluster = None
		if centroids is None:
			cluster = KMeans(n_clusters=self.centroids, random_state=0)
		else:
			cluster = KMeans(n_clusters=centroids, random_state=0)

		cluster.fit(X)
		return cluster

	def set_cluster(self, cluster):
		assert cluster is not None
		self.cluster = cluster

	def predict(self, x, cluster=None):
		x = self.process_raw_data(x)
		y = None
		if cluster is None:
			y = self.cluster.predict(x)
		else:
			y = cluster.predict(x)
		return y