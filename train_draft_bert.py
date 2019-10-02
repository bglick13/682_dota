import pickle
import numpy as np
import pandas as pd
from models.draft_bert import DraftBert, DraftBertTasks
from data_util import parse_draft_orders_for_prediction, parse_graph_for_mlm_prediction
import torch
import networkx as nx


if __name__ == '__main__':
    col_format = ['R_Ban', 'D_Ban', 'R_Ban', 'D_Ban', 'R_Ban', 'D_Ban',
                  'R_Pick', 'D_Pick', 'D_Pick', 'R_Pick',
                  'R_Ban', 'D_Ban', 'R_Ban', 'D_Ban',
                  'D_Pick', 'R_Pick', 'D_Pick', 'R_Pick',
                  'D_Ban', 'R_Ban',
                  'R_Pick', 'D_Pick']

    hero_ids = pd.read_json('const/hero_ids.json', orient='records')
    hero_ids['id'] -= 1  # Make the hero ids 0 indexed
    hero_ids = hero_ids.set_index('id')
    file_name = 'tmp/test_matchups_3132749989.pkl'
    data = nx.read_gpickle(file_name)
    # with open(file_name, 'rb') as f:
    #     # data = pickle.load(f)
    #     data = pickle.load(f)
    data = parse_graph_for_mlm_prediction(data)

    model: DraftBert = DraftBert(embedding_dim=512, n_head=8, n_encoder_layers=4, ff_dim=1024, n_heros=len(hero_ids),
                                 out_ff_dim=512)
    model.cuda()

    train_data = data[:-1000]
    test_data = data[-1000:]
    print(f'Train data shape: {train_data.shape}')
    model.fit(train_data, train_data, DraftBertTasks.DRAFT_PREDICTION, **{'steps': 5000, 'lr': 1.0e-4,
                                                                          'batch_size': 1024})

    all_pred, all_true = [], []
    for i, td in enumerate(test_data):
        td = np.array([td])
        test_masks = model._gen_random_masks(td)
        # test_masks = np.array([np.append([False] * ((i+1) % 22), [True] * (22 - ((i+1) % 22)))])
        pred = model.predict(td, test_masks, DraftBertTasks.DRAFT_PREDICTION)

        actual = td[0]
        actual = np.array([hero_ids.loc[h, 'localized_name'] for h in actual])
        masked = actual.copy()
        masked[test_masks[0]] = None
        predicted = actual.copy()
        pred = pred.argmax(1).detach().cpu().numpy()
        pred = np.array([hero_ids.loc[h, 'localized_name'] for h in pred])
        predicted[test_masks[0]] = pred
        all_pred.extend(pred)
        all_true.extend(actual[test_masks[0]])

        s = pd.DataFrame(index=col_format, data={'Actual': actual, 'Masked': masked, 'Predicted': predicted})
        if i <= 10:
            print(s)
    acc = (np.array(all_pred) == np.array(all_true)).astype(int).mean()
    print(f'test acc: {acc}')