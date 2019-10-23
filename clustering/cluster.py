import os, sys
sys.path.append(os.path.abspath('..'))

import pickle
import numpy as np
import torch

from IPython import embed

from sklearn import metrics
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

nodes = pickle.load( open(os.path.join('clustering','10k_nodes.pickle'), mode='rb'), encoding='bytes')
embeddings = pickle.load( open(os.path.join('clustering','10k_embed.pickle'), mode='rb'), encoding='bytes')

embed()