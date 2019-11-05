import numpy as np
import pandas as pd
import pickle
from models.draft_bert import DraftBert, SelfPlayDataset
from torch import load, device


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
    EPOCHS = 10
    BATCH_SIZE = 64

    with open(memory_file, 'rb') as f:
        mem = pickle.load(f)

    # TODO: I think we can probably convert this to the captiansmodepretrain dataset format and reuse a bunch of code
    model: DraftBert = load('../weights/final_weights/draft_bert_pretrain_captains_mode_2.torch',
                            map_location=device('cpu'))
    # states, actions, values, ids, teams = reconcile_memory(mem, model.mask_idx)
    # df = pd.DataFrame(dict(match_seq_num=ids, hero_id=actions, radiant_win=values, team=teams))
    dataset = SelfPlayDataset(mem)
    model.cuda()
    model.train()
    model.train_from_selfplay(dataset, epochs=10)


if __name__ == '__main__':
    run('../data/self_play/selfplay_1572991709.5282474.pickle')