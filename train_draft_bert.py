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

    col_format = ['R0', 'R1', 'R2', 'R3', 'R4', 'D0', 'D1', 'D2', 'D3', 'D4']

    hero_ids = pd.read_json('const/hero_ids.json', orient='records')
    hero_ids = hero_ids.set_index('id')
    file_name = 'tmp/test_matchups_3132749989.pkl'
    model: DraftBert = DraftBert(embedding_dim=128, n_head=4, n_encoder_layers=3, ff_dim=256, n_heros=len(hero_ids),
                                 out_ff_dim=128)
    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    try:
        with open('data/draft_pretain.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('data/draft_pretain_le.pkl', 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        g = nx.read_gpickle(file_name)
        data, le = parse_graph_for_mlm_prediction(g, hero_ids.index)
        with open('data/draft_pretain.pkl', 'wb') as f:
            pickle.dump(data, f)
        with open('data/draft_pretain_le.pkl', 'wb') as f:
            pickle.dump(le, f)
        del g

    model.cuda()

    train_data = data[:-1000]
    test_data = data[-1000:]
    print(f'Train data shape: {train_data.shape}')
    print(f'Min hero id: {data.min()}, Max hero id: {data.max()}')
    model.fit(train_data, train_data, DraftBertTasks.DRAFT_PREDICTION, **{'steps': int(1e6), 'lr': 1.0e-5,
                                                                          'batch_size': 128, 'mask_pct': 0.1})

    all_pred, all_true = [], []
    for i, td in enumerate(test_data):
        td = np.array([td])
        test_masks = model._gen_random_masks(td, pct=0.1)
        # test_masks = np.array([np.append([False] * ((i+1) % 22), [True] * (22 - ((i+1) % 22)))])
        pred = model.predict(td, test_masks, DraftBertTasks.DRAFT_PREDICTION)

        actual = td[0]
        actual = le.inverse_transform(actual)
        actual = np.array([hero_ids.loc[h, 'localized_name'] for h in actual])
        masked = actual.copy()
        masked[test_masks[0]] = None
        predicted = actual.copy()
        pred = pred.argmax(1).detach().cpu().numpy()

        # Go from dense index prediction to corresponding hero_id
        pred = le.inverse_transform(pred)
        pred = np.array([hero_ids.loc[h, 'localized_name'] for h in pred])
        predicted[test_masks[0]] = pred
        all_pred.extend(pred)
        all_true.extend(actual[test_masks[0]])

        s = pd.DataFrame(index=col_format, data={'Actual': actual, 'Masked': masked, 'Predicted': predicted})
        if i <= 10:
            print(s)
    acc = (np.array(all_pred) == np.array(all_true)).astype(int).mean()
    print(f'test acc: {acc}')
    torch.save(model, 'draft_bert_pretain.torch')