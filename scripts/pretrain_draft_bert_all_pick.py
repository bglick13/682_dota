import pickle
import numpy as np
import pandas as pd
from models.draft_bert import DraftBert, DraftBertTasks, AllPickDataset
from data_util import parse_draft_orders_for_prediction, parse_graph_for_mlm_prediction
import torch
from torch.utils.data import DataLoader
import networkx as nx
import os

if __name__ == '__main__':
    load_dataset = False
    col_format = ['R_Ban', 'D_Ban', 'R_Ban', 'D_Ban', 'R_Ban', 'D_Ban',
                  'R_Pick', 'D_Pick', 'D_Pick', 'R_Pick',
                  'R_Ban', 'D_Ban', 'R_Ban', 'D_Ban',
                  'D_Pick', 'R_Pick', 'D_Pick', 'R_Pick',
                  'D_Ban', 'R_Ban',
                  'R_Pick', 'D_Pick']

    hero_ids = pd.read_json('../const/hero_ids.json', orient='records')
    with open('../weights/kmeans.pickle', 'rb') as f:
        clusterizer = pickle.load(f)
    # hero_ids = hero_ids.set_index('id')
    file_name = '../tmp/test_matchups_3135389989.pkl'
    if os.path.isfile('../data/all_pick_dataset.pickle') and load_dataset:
        with open('../data/all_pick_dataset.pickle', 'rb') as f:
            dataset: AllPickDataset = pickle.load(f)
    else:
        print('creating dataset')
        dataset = AllPickDataset(file_name, hero_ids, mask_pct=.1, test_pct=.15, clusterizer=clusterizer)
        with open('all_pick_dataset_clusters.pickle', 'wb') as f:
            pickle.dump(dataset, f)

    dataset.hero_ids.to_json('../const/draft_bert_clustering_hero_ids.json')

    mask_idx = dataset.MASK
    model: DraftBert = DraftBert(embedding_dim=256, n_head=4, n_encoder_layers=4, ff_dim=256,
                                 n_heros=len(dataset.hero_ids), out_ff_dim=128, mask_idx=mask_idx,
                                 n_clusters=clusterizer.centroids)

    model.next_hero_output.requires_grad = False  # Can't train next hero prediction since all pick data isn't ordered

    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model.cuda()

    model.pretrain_all_pick(dataset, **{'epochs': int(100), 'lr': 1.0e-4, 'batch_size': 1024, 'mask_pct': 0.1})
    torch.save(model, 'draft_bert_pretrain_matching_with_clusters.torch')