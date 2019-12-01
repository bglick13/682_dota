import os
import sys

sys.path.append(os.path.join('..'))

from draft.draft_env import CaptainModeDraft
from models.draft_agent import DraftAgent
import pandas as pd
import numpy as np
import torch
from collections import deque
import pickle
from multiprocessing import Pool, Process
import time
from functools import partial
from const.pro_drafts import drafts


def do_rollout(model, hero_ids, DRAFT, port, verbose=False):
    draft_id = 0

    player = DraftAgent(model=model, pick_first=np.random.choice([True, False]))
    draft = CaptainModeDraft(hero_ids, port)
    state = draft.reset()
    turn = 0
    action = -1

    all_actions = []
    all_states = []
    nn_values = []
    uct_values = []

    while True:
        try:
            npi = draft.draft_order[draft.next_pick_index]
        except IndexError:
            raise IndexError

        if npi < 13:
            if player.pick_first:
                action, uct_value, p, nn_value, leafs = player.act(state, action, num_reads=500, deterministic=True)
                nn_values.append(nn_value)
                uct_values.append(uct_value)
            else:
                try:
                    action = hero_ids.loc[hero_ids['localized_name'] == DRAFT[draft_id], 'model_id'].values[0]
                except:
                    print(DRAFT[draft_id])
                draft_id += 1
        else:
            if player.pick_first:
                try:
                    action = hero_ids.loc[hero_ids['localized_name'] == DRAFT[draft_id], 'model_id'].values[0]
                except:
                    print(DRAFT[draft_id])
                draft_id += 1
            else:
                action, uct_value, p, nn_value, leafs = player.act(state, action, num_reads=500, deterministic=True)
                nn_values.append(nn_value)
                uct_values.append(uct_value)
        all_states.append(state.game_state)
        all_actions.append(action)
        state, value, done = draft.step(action)

        if value == 0:  # Dire victory
            print('Dire victory')
            break
        elif value == 1:
            print('Radiant Victory')
            break
        elif done:
            print('Done but no victory')
            break
        turn += 1

    if (value == 1 and player.pick_first) or (value == 0 and not player.pick_first):
        if player.pick_first:
            print("Agent victory! (pick first)")
        else:
            print("Agent victory! (pick second)")

    else:
        if player.pick_first:
            print('Agent Lost :( (pick first)')
        else:
            print('Agent Lost :( (pick second)')

    all_actions.append(action)
    all_states.append(state.game_state)

    # TODO: I'm really not confident this is right - it's worth double and triple checking
    all_values = [value] * 23
    all_agent_pick_first = [player.pick_first] * 23
    # all_values[[0, 2, 4, 6, 9, 11, 13, 15, 17, 19, 20]] = value
    # all_values[[1, 3, 5, 7, 8, 10, 12, 14, 16, 18, 21]] = 1 - value
    del model
    torch.cuda.empty_cache()
    return dict(all_actions=all_actions, all_states=all_states, all_values=all_values,
                all_agent_pick_first=all_agent_pick_first, nn_values=nn_values, uct_values=uct_values)


if __name__ == '__main__':
    model = torch.load('../data/self_play/memories_for_train_3/new_model.torch', map_location=torch.device('cpu'))
    model.eval()
    model.requires_grad = False

    memory_size = 500000
    n_jobs = 4
    n_games = 200
    port = 13337
    verbose = True
    hero_ids = pd.read_json('../const/draft_bert_clustering_hero_ids.json', orient='records')
    hero_ids['localized_name'] = hero_ids['localized_name'].str.lower()

    memory = deque(maxlen=memory_size)

    times = []
    start = time.time()

    for draft in drafts:
        f = partial(do_rollout, model, hero_ids, draft)
        start_batch = time.time()
        with Pool(n_jobs) as pool:
            results = pool.map_async(f, [port + i for i in range(n_jobs)]).get()
            memory.extend(results)
        times.append(time.time() - start_batch)
        print(f'Finished batch in {times[-1]}s')
    end = time.time()
    with open('../data/self_play/new_model_3_vs_pro_drafts.pickle', 'wb') as f:
        pickle.dump(memory, f)

