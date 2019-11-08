from clustering.kmeans_cluster import KmeansCluster
import pickle


if __name__ == '__main__':
    with open('../data/clustering/100000k_nodes.pickle', 'rb') as f:
        data = pickle.load(f)
    km = KmeansCluster(8)
    km.fit(data)
    with open('../weights/kmeans.pickle', 'wb') as f:
        pickle.dump(km, f)