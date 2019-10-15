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
    col_format = ['R_Ban', 'D_Ban', 'R_Ban', 'D_Ban', 'R_Ban', 'D_Ban',
                  'R_Pick', 'D_Pick', 'D_Pick', 'R_Pick',
                  'R_Ban', 'D_Ban', 'R_Ban', 'D_Ban',
                  'D_Pick', 'R_Pick', 'D_Pick', 'R_Pick',
                  'D_Ban', 'R_Ban',
                  'R_Pick', 'D_Pick']

    hero_ids = pd.read_json('../const/hero_ids.json', orient='records')
    # hero_ids = hero_ids.set_index('id')
    file_name = '../tmp/test_matchups_3135389989.pkl'
    if os.path.isfile('all_pick_dataset.pickle'):
        with open('all_pick_dataset.pickle', 'rb') as f:
            dataset: AllPickDataset = pickle.load(f)
    else:
        dataset = AllPickDataset(file_name, hero_ids, mask_pct=.1, test_pct=.15)
        with open('all_pick_dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)

    dataset.hero_ids.to_json('../const/draft_bert_hero_ids.json')

    mask_idx = dataset.MASK
    model: DraftBert = DraftBert(embedding_dim=256, n_head=4, n_encoder_layers=4, ff_dim=256,
                                 n_heros=len(dataset.hero_ids), out_ff_dim=128, mask_idx=mask_idx)

    model.next_hero_output.requires_grad = False  # Can't train next hero prediction since all pick data isn't ordered

    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model.cuda()

    model.pretrain_all_pick(dataset, **{'epochs': int(1), 'lr': 1.0e-4, 'batch_size': 1024, 'mask_pct': 0.1})

    # all_pred, all_true = [], []
    # for i, td in enumerate(test_data):
    #     td = np.array([td])
    #     test_masks = model._gen_random_masks(td, pct=0.1)
    #     # test_masks = np.array([np.append([False] * ((i+1) % 22), [True] * (22 - ((i+1) % 22)))])
    #     pred = model.predict(td, test_masks, DraftBertTasks.DRAFT_PREDICTION)
    #
    #     actual = td[0]
    #     actual = le.inverse_transform(actual)
    #     actual = np.array([hero_ids.loc[h, 'localized_name'] for h in actual])
    #     masked = actual.copy()
    #     masked[test_masks[0]] = None
    #     predicted = actual.copy()
    #     pred = pred.argmax(1).detach().cpu().numpy()
    #
    #     # Go from dense index prediction to corresponding hero_id
    #     pred = le.inverse_transform(pred)
    #     pred = np.array([hero_ids.loc[h, 'localized_name'] for h in pred])
    #     predicted[test_masks[0]] = pred
    #     all_pred.extend(pred)
    #     all_true.extend(actual[test_masks[0]])
    #
    #     s = pd.DataFrame(index=col_format, data={'Actual': actual, 'Masked': masked, 'Predicted': predicted})
    #     if i <= 10:
    #         print(s)
    # acc = (np.array(all_pred) == np.array(all_true)).astype(int).mean()
    # print(f'test acc: {acc}')
    torch.save(model, 'draft_bert_pretrain_matching.torch')