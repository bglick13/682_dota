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

    model: DraftBert = torch.load('../weights/final_weights/draft_bert_pretrain_all_pick.torch')
    model.has_trained_on_all_pick = True
    assert model.has_trained_on_all_pick, 'Pre-train on all-pick dataset before training on captains mode'
    model.masked_output.requires_grad = False
    model.masked_output.requires_grad = False
    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    file_name = '../tmp/test_draft_order_df_3135389989.pkl'
    with open(file_name, 'rb') as f:
        d = pickle.load(f)
    if os.path.isfile('../data/captains_mode_dataset.pickle') and load_dataset:
        with open('../data/captains_mode_dataset.pickle', 'rb') as f:
            dataset: CaptainsModeDataset = pickle.load(f)
    else:
        dataset = CaptainsModeDataset(file_name, hero_ids=model.hero_ids, label_encoder=model.le, sep=model.sep,
                                      cls=model.cls, mask=model.mask_idx)
        with open('../data/captains_mode_dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)

    model.cuda()

    model.pretrain_captains_mode(dataset, **{'epochs': int(100), 'lr': 1.0e-4, 'batch_size': 64, 'mask_pct': 0.1})
    torch.save(model, '../weights/final_weights/draft_bert_pretrain_captains_mode_2.torch')