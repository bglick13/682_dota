import numpy as np
import pandas as pd
import datetime
import pickle
import os
import networkx as nx


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


def parse_graph_for_mlm_prediction(g: nx.Graph):
    out = []
    for edge in g.edges:
        r_heros = edge