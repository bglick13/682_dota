import numpy as np
import pandas as pd
import datetime
import pickle
import os
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from itertools import product, permutations


def parse_draft_orders_for_prediction(df):
    """
    Output shape (N, seq_length)

    :param df:
    :return:
    """
    out = []
    for key, grp in df.groupby('match_id'):
        heros = grp['hero_id'].values
        heros -= 1  # We want hero_id to be 0 based
        if len(grp) != 22:
            continue
        out.append(heros)
    return np.array(out)


def parse_graph_for_mlm_prediction(g: nx.Graph, hero_ids):
    out = []

    # LabelEncoder makes the hero_ids dense and 0 indexed
    le = LabelEncoder()
    le.fit(hero_ids)
    CLS = len(le.classes_) + 1
    SEP = len(le.classes_) + 2
    MASK = len(le.classes_) + 3
    for edge in tqdm(g.edges(data=True)):
        r = edge[0]
        d = edge[1]
        if 0 in r or 0 in d:  # One of the teams has an invalid hero ID
            continue
        # Start with a CLS token and separate the teams with a SEP token
        heros = np.concatenate(([CLS], le.transform(r), [SEP], le.transform(d), [SEP]))
        for _ in range(len(edge[2]['wins'])):
            out.append(heros)
    return np.array(out), le


def gen_clustering_dataset(g: nx.Graph, model, le, n_nodes):
    nodes = np.array(list(g.nodes()))
    nodes = nodes[np.random.choice(range(len(nodes)), n_nodes)]
    embedded_repr = np.array([le.transform(n) for n in nodes if 0 not in n])
    embedded_repr = model.embed_lineup(embedded_repr)
    embedded_repr = embedded_repr.detach().cpu().numpy()
    return nodes, embedded_repr
