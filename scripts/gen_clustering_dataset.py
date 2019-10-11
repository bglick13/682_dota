import networkx as nx
from data_util import gen_clustering_dataset
import torch
import pickle


if __name__ == '__main__':

    file_name = '../tmp/test_matchups_3135389989.pkl'
    model = torch.load('../draft_bert_pretrain_checkpoint_99999.torch')
    g = nx.read_gpickle(file_name)

    with open('../data/draft_pretrain_le.pkl', 'rb') as f:
        le = pickle.load(f)

    for size in [100000, 1000000]:
        nodes, repr = gen_clustering_dataset(g, model, le, size)

        with open(f'../data/clustering/{size}k_nodes.pickle', 'wb') as f:
            pickle.dump(nodes, f)

        with open(f'../data/clustering/{size}k_embed.pickle', 'wb') as f:
            pickle.dump(repr, f)
    del g