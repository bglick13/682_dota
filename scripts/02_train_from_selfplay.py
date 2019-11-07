import os
import pickle

import numpy as np
from torch import load, device, save

from models.draft_bert import DraftBert, SelfPlayDataset


def reconcile_memory(memory, mask_idx):
    draft_order = np.array([1, 13, 2, 14, 3, 15,
                            4, 16, 5, 17,
                            6, 18, 7, 19,
                            8, 20, 9, 21,
                            10, 22,
                            11, 23])
    draft_order -= 1

    all_outcomes = []
    all_actions = []
    all_states = []
    all_ids = []
    for i, m in enumerate(memory):
        v = m['all_values']
        s = m['all_states']
        a = m['all_actions']
        _a = []

        for pick in draft_order:
            _a.append(a[pick])
        _a.append(mask_idx)
        all_outcomes.extend(v)
        all_actions.extend(_a)
        all_states.extend(s)
        all_ids.extend([i] * len(s))
    all_states = np.array(all_states)
    all_teams = [0] * len(all_states)
    return all_states, all_actions, all_outcomes, all_ids, all_teams


def run(memory_file, ):
    BATCH_SIZE = 128
    N_STEPS = 1000

    mems = []
    for mem in os.listdir(memory_file):
        if mem.endswith('pickle'):
            with open(os.path.join(memory_file, mem), 'rb') as f:
                mem = pickle.load(f)
            mems.append(mem)

    model: DraftBert = load('../weights/final_weights/draft_bert_pretrain_captains_mode_2.torch',
                            map_location=device('cpu'))

    dataset = SelfPlayDataset(*mems)
    total_points_sampled = BATCH_SIZE * N_STEPS
    EPOCHS = total_points_sampled // len(dataset)
    print(f'Dataset size: {len(dataset)}, Epochs: {EPOCHS}')
    model.cuda()
    model.train()
    model.train_from_selfplay(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
    save(model, os.path.join(memory_file, 'new_model.torch'))


if __name__ == '__main__':
    run('../data/self_play/memories_for_train_1')