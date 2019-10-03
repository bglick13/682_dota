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
    CLS = le.classes_.max() + 1
    SEP = le.classes_.max() + 2
    MASK = le.classes_.max() + 3
    for edge in tqdm(g.edges(data=True)):
        if 0 in edge[0] or 0 in edge[1]:  # One of the teams has an invalid hero ID
            continue
        # Start with a CLS token and separate the teams with a SEP token
        heros = np.concatenate((CLS, le.transform(edge[0]), SEP, le.transform(edge[1])))
        for _ in range(len(edge[2]['wins'])):
            out.append(heros)
    return np.array(out), le