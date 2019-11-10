import pickle
import pandas as pd
from models.draft_bert import DraftBert, CaptainsModeDataset, AllPickDataset
import torch

import os

if __name__ == '__main__':
    load_dataset = False
    col_format = ['R_Ban', 'D_Ban', 'R_Ban', 'D_Ban', 'R_Ban', 'D_Ban',
                  'R_Pick', 'D_Pick', 'D_Pick', 'R_Pick',
                  'R_Ban', 'D_Ban', 'R_Ban', 'D_Ban',
                  'D_Pick', 'R_Pick', 'D_Pick', 'R_Pick',
                  'D_Ban', 'R_Ban',
                  'R_Pick', 'D_Pick']

    model: DraftBert = torch.load('../weights/final_weights/draft_bert_pretrain_matching_with_clusters.torch')
    # model.has_trained_on_all_pick = True
    # assert model.has_trained_on_all_pick, 'Pre-train on all-pick dataset before training on captains mode'
    model.masked_output.requires_grad = False
    model.matching_output.requires_grad = False
    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    with open('../weights/kmeans.pickle', 'rb') as f:
        clusterizer = pickle.load(f)

    if os.path.isfile('../data/captains_mode_dataset.pickle') and load_dataset:
        with open('../data/captains_mode_dataset.pickle', 'rb') as f:
            dataset: CaptainsModeDataset = pickle.load(f)
    else:
        file_name = '../tmp/test_draft_order_df_3135389989.pkl'
        dataset = CaptainsModeDataset(file_name, hero_ids=model.hero_ids, label_encoder=model.le, sep=model.sep,
                                      cls=model.cls, mask=model.mask_idx, clusterizer=clusterizer)
        with open('../data/captains_mode_dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)

    model.cuda()

    model.pretrain_captains_mode(dataset, **{'epochs': int(1), 'lr': 1.0e-4, 'batch_size': 64})
    torch.save(model, '../weights/final_weights/draft_bert_pretrain_captains_mode_with_clusters.torch')