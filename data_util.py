import numpy as np
import pandas as pd
import datetime
import pickle
import os
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


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
    for edge in tqdm(g.edges(data=True)):
        heros = np.append(*edge[:2])
        if 0 in heros:
            continue
        heros = le.transform(heros)
        for _ in range(len(edge[2]['wins'])):
            out.append(heros)
    return np.array(out), le