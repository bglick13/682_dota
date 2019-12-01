import copy
import warnings
from enum import Enum
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.functional import F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from clustering.kmeans_cluster import KmeansCluster

warnings.filterwarnings("ignore")


pd.set_option('display.max_rows', 200)
cuda = torch.cuda.is_available()


class DraftBertTasks(Enum):
    DRAFT_PREDICTION = 1
    DRAFT_MATCHING = 2
    WIN_PREDICTION = 3


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape, dtype=bool), k=1).astype('float32')
    return subsequent_mask


class SelfPlayDataset(Dataset):
    def __init__(self, *memories, test_pct=0, clusterizer=None, le=None):
        self.test_pct = test_pct
        self.clusterizer = clusterizer
        self.le = le
        draft_order = np.array([1, 13, 2, 14, 3, 15,
                                4, 16, 5, 17,
                                6, 18, 7, 19,
                                8, 20, 9, 21,
                                10, 22,
                                11, 23])
        # draft_order -= 1

        all_outcomes = []
        all_actions = []
        all_states = []
        all_picks = []
        p1_clusters = []
        p2_clusters = []
        for memory in memories:
            for i, m in enumerate(memory):
                v = m['all_values']
                # Skip games that failed to load and properly record data
                if v[0] == -1:
                    continue
                s = m['all_states']
                if self.clusterizer is not None:
                    p1_cluster = self.le.inverse_transform(s[-1][[4, 5, 8, 9, 11]].astype(int))
                    p1_cluster = self.clusterizer.predict([p1_cluster])[0]

                    p2_cluster = self.le.inverse_transform(s[-1][[16, 17, 20, 21, 23]].astype(int))
                    p2_cluster = self.clusterizer.predict([p2_cluster])[0]
                p1_uct = m['player1_uct_values']
                p2_uct = m['player2_uct_values']
                uct_values = np.append(p1_uct, p2_uct, 0)
                _a = []
                p = []

                for pick in draft_order:
                    subtract = 1 if pick < 13 else 2
                    _a.append(uct_values[pick-subtract])
                    p.append(pick)

                all_outcomes.extend(v[:-1])
                all_actions.extend(_a)
                all_states.extend(s[:-1])
                all_picks.extend(p)
                if self.clusterizer is not None:
                    p1_clusters.extend([p1_cluster] * len(_a))
                    p2_clusters.extend([p2_cluster] * len(_a))
        self.states = np.array(all_states)
        self.actions = np.array(all_actions)
        self.outcomes = np.array(all_outcomes)
        self.pick_indices = np.array(all_picks)
        if self.clusterizer is not None:
            self.p1_clusters = np.array(p1_clusters)
            self.p2_clusters = np.array(p2_clusters)

        if self.test_pct > 0:
            self.test_idxs = np.random.choice(range(len(self.states)), int(len(self.states) * self.test_pct),
                                              replace=False)
            self.train_idxs = np.array(list(set(range(len(self.states))) - set(self.test_idxs)))

        else:
            self.test_idxs = []
            self.train_idxs = np.arange(len(self.states))
        self.train = True

    def __getitem__(self, item):
        s = self.states[item]
        a = self.actions[item]
        v = self.outcomes[item]
        to_predict = self.pick_indices[item]
        if self.clusterizer is not None:
            p1_cluster = self.p1_clusters[item]
            p2_cluster = self.p2_clusters[item]
        else:
            p1_cluster = None
            p2_cluster = None
        return s, a, v, to_predict, p1_cluster, p2_cluster

    def __len__(self):
        return len(self.states)


class CaptainsModeDataset(Dataset):
    def __init__(self, df: Union[pd.DataFrame, str], hero_ids: pd.DataFrame, label_encoder: LabelEncoder=None,
                 sep: int=None, cls: int=None, mask: int=None, test_pct: float = 0, clusterizer: KmeansCluster = None):
        if isinstance(df, pd.DataFrame):
            df = df
        elif isinstance(df, str):
            df = pd.read_pickle(df)

        self.hero_ids = hero_ids
        self.le = label_encoder
        self.test_pct = test_pct
        self.SEP = sep
        self.CLS = cls
        self.MASK = int(mask)
        self.clusterizer = clusterizer

        self.draft_order = np.array([1, 13, 2, 14, 3, 15,
                                     4, 16, 5, 17,
                                     6, 18, 7, 19,
                                     8, 20, 9, 21,
                                     10, 22,
                                     11, 23])

        self.matchups = []
        self.wins = []
        self.fp_clusters = []
        self.sp_clusters = []
        self.mask_idxs = np.append(np.arange(1, 12), np.arange(13, 24))
        for key, grp in df.groupby('match_seq_num'):
            if len(grp) < 22:
                continue
            first_pick_heros = grp['hero_id'].values[[0, 2, 4, 6, 9, 11, 13, 15, 17, 19, 20]]
            second_pick_heros = grp['hero_id'].values[[1, 3, 5, 7, 8, 10, 12, 14, 16, 18, 21]]

            if self.clusterizer is not None:
                # These are different because the steam API orders the picks differently than we do - don't freak out!
                fp_cluster = self.clusterizer.predict([grp['hero_id'].values[[6, 9, 15, 17, 20]]])[0]
                sp_cluster = self.clusterizer.predict([grp['hero_id'].values[[7, 8, 14, 16, 21]]])[0]

            # Transform from Dota hero ids to model hero ids
            if self.le is not None:
                first_pick_heros = self.le.transform(first_pick_heros)
                second_pick_heros = self.le.transform(second_pick_heros)

            heros = np.concatenate((first_pick_heros, second_pick_heros))
            first_pick_win = (grp['team'].values[0] - int(grp['radiant_win'].values[0])) != 0
            self.matchups.append(heros)
            self.wins.append(first_pick_win)
            if self.clusterizer is not None:
                self.fp_clusters.append(fp_cluster)
                self.sp_clusters.append(sp_cluster)

        self.matchups = np.array(self.matchups)
        self.wins = np.array(self.wins)
        self.fp_clusters = np.array(self.fp_clusters)
        self.sp_clusters = np.array(self.sp_clusters)

        if self.test_pct > 0:
            self.test_idxs = np.random.choice(range(len(self.matchups)), int(len(self.matchups) * self.test_pct),
                                              replace=False)
            self.train_idxs = np.array(list(set(range(len(self.matchups))) - set(self.test_idxs)))

        else:
            self.test_idxs = []
            self.train_idxs = np.arange(len(self.matchups))
        self.train = True

    def __getitem__(self, index):
        if self.train:
            index = self.train_idxs[index]
        else:
            index = self.test_idxs[index]
        matchup = self.matchups[index]
        t = copy.deepcopy(matchup)
        matchup = np.tile(matchup, (22, 1))

        m = copy.deepcopy(matchup)
        m = np.hstack((np.ones((22, 1)) * self.CLS,
                       m[:, :12],
                       np.ones((22, 1)) * self.SEP,
                       m[:, 12:],
                       np.ones((22, 1)) * self.SEP))
        mask = np.zeros_like(m)
        for i in range(len(mask)):
            mask[i, self.draft_order[i:]] = 1
        # mask = subsequent_mask(23).squeeze()
        # mask = mask[:-1, 1:]
        # _m = m[:, self.mask_idxs]
        m[mask.astype(bool)] = int(self.MASK)
        m = torch.LongTensor(m)
        if self.clusterizer is not None:
            fp_cluster = self.fp_clusters[index]
            sp_cluster = self.sp_clusters[index]
            clusters = torch.LongTensor([fp_cluster, sp_cluster]).repeat(22).reshape(22, 2)
        else:
            clusters = None

        return (m, t, torch.LongTensor([self.wins[index]]).repeat(22), torch.LongTensor(self.draft_order),
                clusters)

    def __len__(self):
        if self.train:
            return len(self.train_idxs)
        else:
            return len(self.test_idxs)


class AllPickDataset(Dataset):
    def __init__(self, g: Union[nx.Graph, str], hero_ids: pd.DataFrame, test_pct: float = 0, mask_pct=0.1,
                 clusterizer: KmeansCluster = None):
        if isinstance(g, nx.Graph):
            g = g
        elif isinstance(g, str):
            g = nx.read_gpickle(g)

        if 'MASK' not in hero_ids['name']:
            hero_ids = hero_ids.append({'id': -1, 'name': 'MASK'}, ignore_index=True)
        if 'SEP' not in hero_ids['name']:
            hero_ids = hero_ids.append({'id': -2, 'name': 'SEP'}, ignore_index=True)
        if 'CLS' not in hero_ids['name']:
            hero_ids = hero_ids.append({'id': -3, 'name': 'CLS'}, ignore_index=True)

        self.hero_ids = hero_ids
        self.le = LabelEncoder()
        self.hero_ids['model_id'] = self.le.fit_transform(self.hero_ids['id'])
        self.test_pct = test_pct
        self.mask_pct = mask_pct
        self.clusterizer = clusterizer

        self.MASK = self.hero_ids.loc[self.hero_ids['id'] == -1, 'model_id'].values[0]
        self.SEP = self.hero_ids.loc[self.hero_ids['id'] == -2, 'model_id'].values[0]
        self.CLS = self.hero_ids.loc[self.hero_ids['id'] == -3, 'model_id'].values[0]

        print(self.hero_ids)

        self.matchups = []
        self.wins = []
        self.r_clusters = []
        self.d_clusters = []

        for edge in tqdm(g.edges(data=True)):
            r = edge[0]
            d = edge[1]
            if 0 in r or 0 in d:  # One of the teams has an invalid hero ID
                continue
            # Start with a CLS token and separate the teams with a SEP token
            if self.clusterizer is not None:
                r_cluster = self.clusterizer.predict([r])[0]
                d_cluster = self.clusterizer.predict([d])[0]
            r = self.le.transform(r)
            d = self.le.transform(d)

            heros = np.concatenate(([self.CLS],  # Always start with the CLS token - index [0]
                                    # Radiant - starts at index 1
                                    np.ones(3) * self.MASK,  # First wave of 3 bans - indices [1, 2, 3]
                                    r[:2],  # First two picks - indices [4, 5]
                                    np.ones(2) * self.MASK,  # Two more bans - indices [6, 7]
                                    r[2:4],  # Two more picks - indices [8, 9]
                                    np.ones(1) * self.MASK,  # Final ban - index [10]
                                    r[4:5],  # Final pick - index [11]
                                    [self.SEP],  # Index [12]

                                    # Dire - starts at index 13
                                    np.ones(3) * self.MASK,  # First wave of 3 bans - indices [13, 14, 15]
                                    d[:2],  # First two picks - indices [16, 17]
                                    np.ones(2) * self.MASK,  # Two more bans - indices [18, 19]
                                    d[2:4],  # Two more picks - indices [20, 21]
                                    np.ones(1) * self.MASK,  # Final ban - index [22]
                                    d[4:5],  # Final pick - index [23]
                                    [self.SEP]  # Index [24])
                                    ))
            for _, w in enumerate(edge[2]['wins']):
                w = self.le.transform(w)
                self.matchups.append(heros)
                if self.clusterizer is not None:
                    self.r_clusters.append(r_cluster)
                    self.d_clusters.append(d_cluster)
                # 1 if Radiant victory, 0 if Dire victory
                if np.sum(w == r) == 5:
                    self.wins.append(1)
                else:
                    self.wins.append(0)
        del g
        self.matchups = np.array(self.matchups)
        self.wins = np.array(self.wins)

        if self.test_pct > 0:
            self.test_idxs = np.random.choice(range(len(self.matchups)), int(len(self.matchups) * self.test_pct), replace=False)
            self.train_idxs = np.array(list(set(range(len(self.matchups))) - set(self.test_idxs)))

        else:
            self.test_idxs = []
            self.train_idxs = np.arange(len(self.matchups))
        self.train = True

    def _gen_random_masks(self):
        """

        :param x: shape (batch_size, sequence_length, 1)
        :param pct:
        :return:
        """
        n_masked_idx = int(10 * self.mask_pct)
        mask = np.append([1] * n_masked_idx, [0] * (10 - n_masked_idx))
        mask = np.random.permutation(mask)
        mask = np.concatenate((np.zeros(1),
                               np.zeros(3),
                               mask[:2],
                               np.zeros(2),
                               mask[2:4],
                               np.zeros(1),
                               mask[4:5],
                               np.zeros(1),
                               np.zeros(3),
                               mask[5:7],
                               np.zeros(2),
                               mask[7:9],
                               np.zeros(1),
                               mask[9:],
                               np.zeros(1)))

        mask = torch.BoolTensor(mask)
        return mask

    def __getitem__(self, index):
        mask = self._gen_random_masks()
        if self.train:
            index = self.train_idxs[index]
        else:
            index = self.test_idxs[index]
        m = self.matchups[index]
        r = np.random.permutation(m[[4, 5, 8, 9, 11]])
        d = np.random.permutation(m[[16, 17, 20, 21, 23]])

        if self.clusterizer is not None:
            r_cluster = self.r_clusters[index]
            d_cluster = self.d_clusters[index]
        else:
            r_cluster = None
            d_cluster = None

        m[[4, 5, 8, 9, 11]] = r
        m[[16, 17, 20, 21, 23]] = d

        return (torch.LongTensor(m),
                torch.LongTensor([self.wins[index]]),
                mask,
                torch.LongTensor([r_cluster, d_cluster]))

    def __len__(self):
        if self.train:
            return len(self.train_idxs)
        else:
            return len(self.test_idxs)


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def swish(x):
    return F.sigmoid(x) * x


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.sigmoid(x) * x


class DraftBert(torch.nn.Module):
    def __init__(self, embedding_dim, ff_dim, n_head, n_encoder_layers, n_heros, out_ff_dim, mask_idx, n_clusters=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers
        self.n_heros = n_heros
        self.n_clusters = n_clusters

        self.encoder_layer = torch.nn.TransformerEncoderLayer(embedding_dim, n_head, dim_feedforward=ff_dim, dropout=0.2)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)

        # Masked output layers
        self.masked_output_hidden = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim),
                                                 torch.nn.LayerNorm(out_ff_dim),
                                                 Swish())
        self.masked_output_out = torch.nn.Linear(out_ff_dim, n_heros)
        self.masked_output = torch.nn.Sequential(self.masked_output_hidden, self.masked_output_out)

        # Matching classifier layer - Only used for pretraining
        self.matching_output = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim),
                                                   torch.nn.LayerNorm(out_ff_dim),
                                                   Swish(),
                                                   torch.nn.Linear(out_ff_dim, 2))
        # self.matching_layer = torch.nn.Linear(embedding_dim, 2)

        # Win classifier/Value head
        self.win_output = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim),
                                                   torch.nn.LayerNorm(out_ff_dim),
                                                   Swish(),
                                                   torch.nn.Linear(out_ff_dim, 2))
        # Next hero prediction/Policy head
        self.next_hero_output_hidden = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim),
                                                   torch.nn.LayerNorm(out_ff_dim),
                                                   Swish())
        self.next_hero_out = torch.nn.Linear(out_ff_dim, n_heros)
        self.next_hero_output = torch.nn.Sequential(self.next_hero_output_hidden, self.next_hero_out)

        # Cluster related layers
        self.cluster_output_hidden = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim),
                                                         torch.nn.LayerNorm(out_ff_dim),
                                                         torch.nn.ReLU())
        self.cluster_out = torch.nn.Linear(out_ff_dim, n_clusters)
        self.cluster_output = torch.nn.Sequential(self.cluster_output_hidden, self.cluster_out)
        self.friendly_cluster_update = torch.nn.Sequential(torch.nn.Linear(out_ff_dim, out_ff_dim),
                                                           torch.nn.LayerNorm(out_ff_dim))
        self.opponent_cluster_update = torch.nn.Sequential(torch.nn.Linear(out_ff_dim, out_ff_dim),
                                                           torch.nn.LayerNorm(out_ff_dim))

        dictionary_size = n_heros
        self.hero_embeddings = torch.nn.Embedding(dictionary_size, embedding_dim, padding_idx=int(mask_idx))
        self.pe = PositionalEncoding(embedding_dim, 0, max_len=25)

        self.mask_idx = int(mask_idx)
        self.hero_ids = None
        self.sep = None
        self.cls = None
        self.le = None
        self.has_trained_on_all_pick = False
        self.has_trained_on_captains_mode = False

    def get_cluster_predictions(self, src, mask=None):
        if self.n_clusters is None:
            return None, None, [None, None]
        if mask is not None:
            src[mask] = self.mask_idx
        first_to_pick_embeddings = src[:, [4, 5, 8, 9, 11], :].sum(1)
        second_to_pick_embeddings = src[:, [16, 17, 20, 21, 23], :].sum(1)

        first_to_pick_hidden = self.cluster_output_hidden(first_to_pick_embeddings)
        second_to_pick_hidden = self.cluster_output_hidden(second_to_pick_embeddings)

        first_to_pick_cluster = self.cluster_out(first_to_pick_hidden)
        second_to_pick_cluster = self.cluster_out(second_to_pick_hidden)
        return (first_to_pick_cluster, second_to_pick_cluster,
                torch.cat([first_to_pick_hidden, second_to_pick_hidden]).view(2, first_to_pick_hidden.shape[0],
                                                                              first_to_pick_hidden.shape[1]).permute(1, 0, 2))

    def embed_lineup(self, lineup):
        if isinstance(lineup, (list, np.ndarray)):
            lineup = torch.LongTensor(lineup)
        if len(lineup.shape) == 1:
            lineup = lineup.unsqueeze(0)
        if cuda:
            lineup = lineup.cuda()
        return self.hero_embeddings(lineup)

    def get_attn_maps(self, src: torch.LongTensor, mask: torch.BoolTensor = None):
        if mask is not None:
            src[mask] = self.mask_idx
        src = self.hero_embeddings(src)
        src = src + np.sqrt(self.embedding_dim)
        src = self.pe(src)

        # Encoder expects shape (seq_length, batch_size, embedding_dim)
        src = src.permute(1, 0, 2)
        # Then we pass it through the encoder stack
        # out = self.encoder(src, src_key_padding_mask=mask, src_mask)
        attn_maps = []
        for i in range(self.encoder.num_layers):
            output = self.encoder.layers[i].self_attn(src, src, src, need_weights=True)
            attn_maps.append(output)
        return attn_maps

    def forward(self, src: torch.LongTensor, mask: torch.BoolTensor = None):
        """

        :param src: shape (batch_size, seq_length, 1) a sequence of hero_ids representing a draft
        :param tgt: shape (batch_size, seq_length, 1) a sequence of hero_ids representing a draft
        :param mask: shape (batch_size, seq_length) a boolean sequence of which indexes in the draft should be masked
        :return: shape (batch_size, seq_length, embedding_dim) encoded representation of the sequence
        """

        # First, we encode the src sequence into the latent hero representation
        if mask is not None:
            src[mask] = self.mask_idx
        # if cuda:
        #     src = src.cuda()  # Set the masked values to the embedding pad idx
        src = self.hero_embeddings(src)
        # TODO: This should be a multiplication
        src = src + np.sqrt(self.embedding_dim)
        src = self.pe(src)

        # Encoder expects shape (seq_length, batch_size, embedding_dim)
        src = src.permute(1, 0, 2)
        # Then we pass it through the encoder stack
        # out = self.encoder(src, src_key_padding_mask=mask, src_mask)
        out = self.encoder(src)
        # Encoder outputs shape (seq_length, batch_size, embedding_dim)
        out = out.permute(1, 0, 2)

        return out

    def get_masked_output(self, x, friendly_cluster_h=None, opponent_cluster_h=None):
        if self.n_clusters is None:
            return self.masked_output(x)
        else:
            masked_hero_h = self.masked_output_hidden(x)
            # TODO: need to call self.friendly_cluster_update(friendly_cluster_h)
            masked_hero_h += friendly_cluster_h
            masked_hero_h += opponent_cluster_h
            masked_hero_h = F.relu(masked_hero_h)
            return self.masked_output_out(masked_hero_h)

    def get_matching_output(self, x):
        return self.matching_output(x)

    def get_next_hero_output(self, x, friendly_cluster_h=None, opponent_cluster_h=None):
        if self.n_clusters is None or friendly_cluster_h is None or opponent_cluster_h is None:
            return self.next_hero_output(x)
        else:
            next_hero_h = self.next_hero_output_hidden(x)
            # TODO: need to call self.friendly_cluster_update(friendly_cluster_h)
            next_hero_h += friendly_cluster_h
            next_hero_h += opponent_cluster_h
            next_hero_h = F.relu(next_hero_h)
            return self.next_hero_out(next_hero_h)

    def get_win_output(self, x):
        return self.win_output(x)

    def pretrain_all_pick(self, dataset: AllPickDataset, **train_kwargs):
        self.has_trained_on_all_pick = True
        self.hero_ids = dataset.hero_ids
        self.cls = dataset.CLS
        self.sep = dataset.SEP
        self.le = dataset.le
        lr = train_kwargs.get('lr', 0.001)
        batch_size = train_kwargs.get('batch_size', 512)
        epochs = train_kwargs.get('epochs', 100)
        mask_pct = train_kwargs.get('mask_pct', 0.1)
        print_iter = train_kwargs.get('print_iter', 100)
        save_iter = train_kwargs.get('save_iter', 100000)
        test = train_kwargs.get('test', False)
        if test:
            dataset.train = False
            self.eval()
        else:
            self.train()

        dataset.mask_pct = mask_pct
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        if not test:
            opt = torch.optim.Adam([{'params': self.encoder.parameters()},
                                {'params': self.win_output.parameters()},
                                {'params': self.masked_output.parameters()},
                                {'params': self.matching_output.parameters()},
                                {'params': self.cluster_output.parameters()},
                                {'params': self.friendly_cluster_update.parameters()},
                                {'params': self.opponent_cluster_update.parameters()},
                                {'params': self.hero_embeddings.parameters()}], lr=lr)
        mask_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        matching_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        win_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        cluster_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        hist = dict(epoch=[], step=[], masked_hero_acc=[], top_5_acc=[], cluster_acc=[], win_acc=[], loss=[])
        for epoch in tqdm(range(epochs)):
            for i, batch in tqdm(enumerate(dataloader)):
                if not test:
                    opt.zero_grad()

                src_batch = batch[0]
                tgt_batch = copy.deepcopy(src_batch)
                win_batch = batch[1]
                mask_batch = batch[2]
                cluster_batch = batch[3]

                # Randomly shuffle the matchups for half the batch
                is_correct_matchup = np.random.choice([0, 1], src_batch.shape[0])
                shuffled_lineups = src_batch[is_correct_matchup == 0, :][:, [16, 17, 20, 21, 23]]
                shuffled_lineups = shuffled_lineups[torch.randperm(shuffled_lineups.size()[0])]
                src_batch[is_correct_matchup == 0, :][:, [16, 17, 20, 21, 23]] = shuffled_lineups
                if cuda:
                    src_batch = src_batch.cuda()
                    mask_batch = mask_batch.cuda()

                out = self.forward(src_batch, mask_batch)  # -> shape (batch_size, sequence_length, embedding_dim)
                if self.n_clusters is not None:
                    cluster_out = self.get_cluster_predictions(out, mask_batch)

                to_predict = out[mask_batch]
                if self.n_clusters is not None:
                    picking_team = (mask_batch.argmax(-1) > 12).long()
                    friendly_cluster_hs = cluster_out[2][range(tgt_batch.shape[0]), picking_team]
                    opponent_cluster_hs = cluster_out[2][range(tgt_batch.shape[0]), 1 - picking_team]
                else:
                    friendly_cluster_hs = None
                    opponent_cluster_hs = None
                mask_pred = self.get_masked_output(to_predict, friendly_cluster_hs, opponent_cluster_hs)

                if self.n_clusters is not None:
                    cluster_pred = torch.cat((cluster_out[0], cluster_out[1]))
                    cluster_tgt = torch.cat((cluster_batch[:, 0], cluster_batch[:, 1])).cuda()
                    cluster_loss_batch = cluster_loss(cluster_pred, cluster_tgt)

                mask_tgt_batch = tgt_batch[mask_batch]
                if cuda:
                    mask_tgt_batch = mask_tgt_batch.cuda()
                mask_batch_loss = mask_loss(mask_pred, mask_tgt_batch)

                is_correct_pred = self.get_matching_output(out[:, 0, :])
                is_correct_matchup = torch.LongTensor(is_correct_matchup)
                if cuda:
                    is_correct_matchup = is_correct_matchup.cuda()
                is_correct_loss = matching_loss(is_correct_pred, is_correct_matchup)

                win_pred = self.get_win_output(out[:, 0, :])
                if cuda:
                    win_batch = win_batch.cuda()
                batch_win_loss = win_loss(win_pred, win_batch.squeeze())

                batch_loss = (mask_batch_loss + is_correct_loss + batch_win_loss)
                if self.n_clusters is None:
                    batch_loss /= 3
                else:
                    batch_loss += cluster_loss_batch
                    batch_loss /= 4

                if not test:
                    batch_loss.backward()
                    opt.step()

                if i == 0 or (i + 1) % print_iter == 0:
                    batch_acc = (
                                mask_pred.detach().cpu().numpy().argmax(1) == mask_tgt_batch.detach().cpu().numpy()).astype(
                        int).mean()
                    top_5_pred = np.argsort(mask_pred.detach().cpu().numpy(), axis=1)[:, -5:]
                    top_5_acc = np.array(
                        [t in p for t, p in zip(mask_tgt_batch.detach().cpu().numpy(), top_5_pred)]).astype(int).mean()

                    matching_acc = (is_correct_pred.detach().cpu().numpy().argmax(
                        1) == is_correct_matchup.detach().cpu().numpy()).astype(int).mean()

                    win_acc = (win_pred.detach().cpu().numpy().argmax(
                        1) == win_batch.squeeze().detach().cpu().numpy()).astype(int).mean()

                    if self.n_clusters is not None:
                        cluster_acc = (cluster_pred.detach().cpu().numpy().argmax(
                        1) == cluster_tgt.squeeze().detach().cpu().numpy()).astype(int).mean()
                    else:
                        cluster_acc = -1
                    hist['epoch'].append(epoch)
                    hist['step'].append(i)
                    hist['masked_hero_acc'].append(batch_acc)
                    hist['cluster_acc'].append(cluster_acc)
                    hist['win_acc'].append(win_acc)
                    hist['top_5_acc'].append(top_5_acc)
                    hist['loss'].append(batch_loss.detach().cpu().numpy())
                    print(
                        f'Epoch: {epoch}, Step: {i}, Loss: {batch_loss}, Acc: {batch_acc}, Top 5 Acc: {top_5_acc},'
                        f'Matching Acc: {matching_acc}, Win Acc: {win_acc}, Cluster Acc: {cluster_acc}')
            if not test:
                torch.save(self, f'../weights/checkpoints/draft_bert_pretrain_allpick_checkpoint_{epoch}.torch')
        return hist

    def train_from_selfplay(self, dataset: SelfPlayDataset, **train_kwargs):
        self.train()
        lr = train_kwargs.get('lr', 0.001)
        batch_size = train_kwargs.get('batch_size', 512)
        epochs = train_kwargs.get('epochs', 100)
        print_iter = train_kwargs.get('print_iter', 100)
        max_steps = train_kwargs.get('steps', np.inf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        opt = torch.optim.Adam([{'params': self.encoder.parameters()},
                                {'params': self.win_output.parameters()},
                                {'params': self.next_hero_output.parameters()},
                                {'params': self.cluster_output.parameters()},
                                {'params': self.friendly_cluster_update.parameters()},
                                {'params': self.opponent_cluster_update.parameters()},
                                {'params': self.hero_embeddings.parameters()}], lr=lr)

        next_hero_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        win_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        cluster_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        total_steps = 0
        for epoch in tqdm(range(epochs)):
            for i, batch in tqdm(enumerate(dataloader)):
                if total_steps > max_steps:
                    break
                opt.zero_grad()

                state_batch = batch[0]
                action_batch = batch[1]
                win_batch = batch[2]
                to_predict = batch[3]
                p1_cluster_batch = batch[4]
                p2_cluster_batch = batch[5]

                state_batch = torch.LongTensor(state_batch.reshape(-1, 25).long())
                action_batch = torch.LongTensor(action_batch.argmax(-1).long())
                win_batch = torch.LongTensor(win_batch.reshape(-1, 1).long())
                to_predict = to_predict.reshape(-1, 1).long()

                # 1 if Dire, else Radiant
                picking_team = (to_predict > 12).squeeze().long()
                p1_cluster_batch = p1_cluster_batch.reshape(-1, 1).long()
                p2_cluster_batch = p2_cluster_batch.reshape(-1, 1).long()

                if cuda:
                    state_batch = state_batch.cuda()
                    action_batch = action_batch.cuda()
                    win_batch = win_batch.cuda()
                    p1_cluster_batch = p1_cluster_batch.cuda()
                    p2_cluster_batch = p2_cluster_batch.cuda()

                out = self.forward(state_batch)  # -> shape (batch_size, sequence_length, embedding_dim)
                if self.n_clusters is not None:
                    cluster_out = self.get_cluster_predictions(out)
                    # [batch_size, [Radiant, Dire], hidden_size]
                    friendly_cluster_hs = cluster_out[2][range(state_batch.shape[0]), picking_team]
                    opponent_cluster_hs = cluster_out[2][range(state_batch.shape[0]), 1 - picking_team]
                    cluster_pred = torch.cat((cluster_out[0], cluster_out[1]))
                    cluster_tgt = torch.cat((p1_cluster_batch, p2_cluster_batch)).cuda()
                    cluster_loss_batch = cluster_loss(cluster_pred, cluster_tgt.squeeze())
                else:
                    friendly_cluster_hs = None
                    opponent_cluster_hs = None
                to_predict = out[range(len(out)), to_predict.squeeze()]
                mask_pred = self.get_next_hero_output(to_predict, friendly_cluster_hs, opponent_cluster_hs)

                # mask_pred = F.log_softmax(mask_pred, -1)
                # mask_tgt = F.softmax(action_batch, -1)
                mask_batch_loss = next_hero_loss(mask_pred, action_batch.squeeze())

                win_pred = self.get_win_output(out[:, 0, :])
                batch_win_loss = win_loss(win_pred, win_batch.squeeze())

                batch_loss = (mask_batch_loss + batch_win_loss)
                if self.n_clusters is not None:
                    batch_loss += cluster_loss_batch
                    batch_loss /= 3
                else:
                    batch_loss /= 2
                batch_loss.backward()
                opt.step()

                if i == 0 or (i + 1) % print_iter == 0:
                    batch_acc = (
                            mask_pred.detach().cpu().numpy().argmax(1) == action_batch.detach().cpu().numpy()).astype(
                        int).mean()
                    top_5_pred = np.argsort(mask_pred.detach().cpu().numpy(), axis=1)[:, -5:]
                    top_5_true = action_batch.detach().cpu().numpy()
                    top_5_acc = np.array(
                        [t in p for t, p in zip(top_5_true, top_5_pred)]).astype(int).mean()

                    win_acc = (win_pred.detach().cpu().numpy().argmax(
                        1) == win_batch.squeeze().detach().cpu().numpy()).astype(int).mean()

                    if self.n_clusters is not None:
                        cluster_acc = (cluster_pred.detach().cpu().numpy().argmax(
                        1) == cluster_tgt.squeeze().detach().cpu().numpy()).astype(int).mean()
                    else:
                        cluster_acc = -1

                    print(
                        f'Epoch: {epoch}, Step: {i}, Loss: {batch_loss} (Hero: {mask_batch_loss}, Cluster: {cluster_loss_batch}, Win: {batch_win_loss}), Acc: {batch_acc}, Top 5 Acc: {top_5_acc}, '
                        f'Win Acc: {win_acc}, Cluster Acc: {cluster_acc}')
                total_steps += 1
            torch.save(self, f'../weights/checkpoints/draft_bert_selfplay_checkpoint_{epoch}.torch')

    def pretrain_captains_mode(self, dataset: CaptainsModeDataset, **train_kwargs):
        lr = train_kwargs.get('lr', 0.001)
        batch_size = train_kwargs.get('batch_size', 512)
        epochs = train_kwargs.get('epochs', 100)
        print_iter = train_kwargs.get('print_iter', 100)
        test = train_kwargs.get('test', False)
        if test:
            self.eval()
            self.requires_grad = False
        else:
            self.train()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        opt = torch.optim.Adam([{'params': self.encoder.parameters()},
                                {'params': self.win_output.parameters()},
                                {'params': self.next_hero_output.parameters()},
                                {'params': self.matching_output.parameters()},
                                {'params': self.cluster_output.parameters()},
                                {'params': self.friendly_cluster_update.parameters()},
                                {'params': self.opponent_cluster_update.parameters()},
                                {'params': self.hero_embeddings.parameters()}], lr=lr)
        next_hero_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        win_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        cluster_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        hist = dict(epoch=[], step=[], masked_hero_acc=[], top_5_acc=[], cluster_acc=[], win_acc=[], loss=[])
        for epoch in tqdm(range(epochs)):
            for i, batch in tqdm(enumerate(dataloader)):
                if not test:
                    opt.zero_grad()

                src_batch = batch[0]
                tgt_batch = batch[1]
                win_batch = batch[2]
                to_predict = batch[3]
                cluster_batch = batch[4]

                src_batch = src_batch.reshape(-1, 25)
                tgt_batch = tgt_batch.reshape(-1, 1)
                win_batch = win_batch.reshape(-1, 1)
                to_predict = to_predict.reshape(-1, 1)
                picking_team = (to_predict > 12).squeeze().long()
                cluster_batch = cluster_batch.reshape(-1, 2).long()

                if cuda:
                    src_batch = src_batch.cuda()
                    tgt_batch = tgt_batch.cuda()
                    win_batch = win_batch.cuda()
                    cluster_batch = cluster_batch.cuda()

                out = self.forward(src_batch)  # -> shape (batch_size, sequence_length, embedding_dim)
                if self.n_clusters is not None:
                    cluster_out = self.get_cluster_predictions(out)
                    friendly_cluster_hs = cluster_out[2][range(tgt_batch.shape[0]), picking_team]
                    opponent_cluster_hs = cluster_out[2][range(tgt_batch.shape[0]), 1 - picking_team]
                    cluster_pred = torch.cat((cluster_out[0], cluster_out[1]))
                    cluster_tgt = torch.cat((cluster_batch[:, 0], cluster_batch[:, 1])).cuda()
                    cluster_loss_batch = cluster_loss(cluster_pred, cluster_tgt)
                else:
                    friendly_cluster_hs = None
                    opponent_cluster_hs = None

                to_predict = out[range(len(out)), to_predict.squeeze()]

                mask_pred = self.get_next_hero_output(to_predict, friendly_cluster_hs, opponent_cluster_hs)

                mask_batch_loss = next_hero_loss(mask_pred, tgt_batch.squeeze())

                win_pred = self.get_win_output(out[:, 0, :])
                batch_win_loss = win_loss(win_pred, win_batch.squeeze())

                batch_loss = (mask_batch_loss + batch_win_loss)
                if self.n_clusters is None:
                    batch_loss /= 2
                else:
                    batch_loss += cluster_loss_batch
                    batch_loss /= 3
                if not test:
                    batch_loss.backward()
                    opt.step()

                if i == 0 or (i + 1) % print_iter == 0:
                    batch_acc = (
                                mask_pred.detach().cpu().numpy().argmax(1) == tgt_batch.detach().cpu().numpy()).astype(
                        int).mean()
                    top_5_pred = np.argsort(mask_pred.detach().cpu().numpy(), axis=1)[:, -5:]
                    top_5_acc = np.array(
                        [t in p for t, p in zip(tgt_batch.detach().cpu().numpy(), top_5_pred)]).astype(int).mean()

                    win_acc = (win_pred.detach().cpu().numpy().argmax(
                        1) == win_batch.squeeze().detach().cpu().numpy()).astype(int).mean()

                    if self.n_clusters is not None:
                        cluster_acc = (cluster_pred.detach().cpu().numpy().argmax(
                        1) == cluster_tgt.squeeze().detach().cpu().numpy()).astype(int).mean()
                    else:
                        cluster_acc = -1
                    hist['epoch'].append(epoch)
                    hist['step'].append(i)
                    hist['masked_hero_acc'].append(batch_acc)
                    hist['cluster_acc'].append(cluster_acc)
                    hist['win_acc'].append(win_acc)
                    hist['top_5_acc'].append(top_5_acc)
                    hist['loss'].append(batch_loss.detach().cpu().numpy())
                    print(
                        f'Epoch: {epoch}, Step: {i}, Loss: {batch_loss}, Acc: {batch_acc}, Top 5 Acc: {top_5_acc},'
                        f'Win Acc: {win_acc}, Cluster Acc: {cluster_acc}')
            torch.save(self, f'../weights/checkpoints/draft_bert_pretrain_captains_mode_checkpoint_{epoch}.torch')
        return hist

    # def fit(self, src: torch.LongTensor, tgt: torch.LongTensor, task: DraftBertTasks, **train_kwargs):
    #     """
    #
    #     :param src: shape (N, sequence_length, 1) where N is the size of the dataset
    #     :param tgt: shape (N, sequence_length, 1) where N is the size of the dataset
    #     :param task: Either draft prediction (fill in masked values) or draft matching (are these 2 teams from the same draft)
    #     :param train_kwargs: lr, batch_size, steps
    #     :return:
    #     """
    #     if isinstance(src, (list, np.ndarray)):
    #         src = torch.LongTensor(src)
    #     if isinstance(tgt, (list, np.ndarray)):
    #         tgt = torch.LongTensor(tgt)
    #     # Don't forget to set to train mode
    #     self.train()
    #     # self.cuda()
    #     lr = train_kwargs.get('lr', 0.001)
    #     batch_size = train_kwargs.get('batch_size', 512)
    #     steps = train_kwargs.get('steps', 100)
    #     mask_pct = train_kwargs.get('mask_pct', 0.1)
    #     print_iter = train_kwargs.get('print_iter', 100)
    #     save_iter = train_kwargs.get('save_iter', 100000)
    #
    #     if task == DraftBertTasks.DRAFT_PREDICTION:
    #         opt = torch.optim.Adam(self.parameters(), lr=lr)
    #         N = src.shape[0]
    #         mask_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    #         matching_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    #         for step in tqdm(range(steps)):
    #             opt.zero_grad()
    #             idxs = np.random.choice(N, batch_size)
    #
    #             # Sample a batch of matchups
    #             src_batch, tgt_batch = src[idxs], tgt[idxs]
    #
    #             # Randomly shuffle the order for each team to avoid sorted bias
    #             src_batch_r = src_batch[:, 1:6]
    #             src_batch_r = src_batch_r[:, torch.randperm(5)]
    #             src_batch_d = src_batch[:, 7:12]
    #             src_batch_d = src_batch_d[:, torch.randperm(5)]
    #             src_batch[:, 1:6] = src_batch_r
    #             src_batch[:, 7:12] = src_batch_d
    #
    #             # Randomly shuffle the matchups for half the batch
    #             is_correct_matchup = np.random.choice([0, 1], batch_size)
    #             shuffled_lineups = src_batch[is_correct_matchup == 0, 7:12]
    #             shuffled_lineups = shuffled_lineups[torch.randperm(shuffled_lineups.size()[0])]
    #             src_batch[is_correct_matchup == 0, 7:12] = shuffled_lineups
    #
    #             # Generate masks for random heros
    #             masks = self._gen_random_masks(src_batch, mask_pct)
    #             if cuda:
    #                 src_batch = src_batch.cuda()
    #                 masks = masks.cuda()
    #
    #             out = self.forward(src_batch, masks)  # -> shape (batch_size, sequence_length, embedding_dim)
    #             to_predict = out[masks]
    #             mask_pred = self.get(to_predict)
    #             mask_tgt_batch = tgt_batch[masks].cuda()
    #             mask_batch_loss = mask_loss(mask_pred, mask_tgt_batch)
    #
    #             is_correct_pred = self.get_matching_output(out[:, 0, :])
    #             is_correct_matchup = torch.LongTensor(is_correct_matchup)
    #             if cuda:
    #                 is_correct_matchup = is_correct_matchup.cuda()
    #             is_correct_loss = matching_loss(is_correct_pred, is_correct_matchup)
    #             batch_loss = (mask_batch_loss + is_correct_loss) / 2.
    #             batch_loss.backward()
    #             opt.step()
    #
    #             if step == 0 or (step+1) % print_iter == 0:
    #                 batch_acc = (mask_pred.detach().cpu().numpy().argmax(1) == mask_tgt_batch.detach().cpu().numpy()).astype(int).mean()
    #                 top_5_pred = np.argsort(mask_pred.detach().cpu().numpy(), axis=1)[:, -5:]
    #                 top_5_acc = np.array([t in p for t, p in zip(mask_tgt_batch.detach().cpu().numpy(), top_5_pred)]).astype(int).mean()
    #                 matching_acc = (is_correct_pred.detach().cpu().numpy().argmax(1) == is_correct_matchup.detach().cpu().numpy()).astype(int).mean()
    #
    #                 print(f'Step: {step}, Loss: {batch_loss}, Acc: {batch_acc}, Top 5 Acc: {top_5_acc}, Matching Acc: {matching_acc}')
    #             if (step+1) % save_iter == 0:
    #                 torch.save(self, f'draft_bert_pretrain_checkpoint_{step}.torch')

    def predict(self, src: torch.LongTensor, mask: torch.BoolTensor, task: DraftBertTasks,
                **predict_kwargs):

        if isinstance(src, (list, np.ndarray)):
            src = torch.LongTensor(src)
        if isinstance(mask, (list, np.ndarray)):
            mask = torch.BoolTensor(mask)
        self.eval()
        if task == DraftBertTasks.DRAFT_PREDICTION:
            if cuda:
                src = src.cuda()
                mask = mask.cuda()
            out = self.forward(src, mask)  # -> shape (batch_size, sequence_length, embedding_dim)
            to_predict = out[mask]
            pred = self.get_masked_output(to_predict)

            # pred = self.dense_layer(to_predict)
            return pred