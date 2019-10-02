import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
pd.set_option('display.max_columns', 9999)

if __name__ == '__main__':
    with open('tmp/test_matchups_3125230594.pkl', 'rb') as f:
        g = pickle.load(f)
    with open('test_draft_order_df_3124306594.pkl', 'rb') as f:
        df = pickle.load(f)
    n_matchups = []
    for edge in tqdm(list(g.edges._adjdict.items())):
        results = edge[1]
        edge = results[list(results.keys())[0]]
        n = len(edge['wins'])
        n_matchups.append(n)
    df['start_time'] = pd.to_datetime(df['start_time'])
    print(df.shape)
    print(df.describe())