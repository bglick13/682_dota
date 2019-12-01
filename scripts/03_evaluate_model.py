import pickle
import time
from collections import deque
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from torch import load, device
from torch.cuda import empty_cache

from draft.draft_env import CaptainModeDraft
from models.draft_agent import DraftAgent, DraftBert


def do_rollout(old_model, new_model, hero_ids, port, verbose=False):
    player_1_pick_first = np.random.choice([True, False])
    player1: DraftAgent = DraftAgent(model=old_model, pick_first=player_1_pick_first)
    player2: DraftAgent = DraftAgent(model=new_model, pick_first=not player_1_pick_first)
    draft = CaptainModeDraft(hero_ids, port)
    state = draft.reset()
    turn = 0
    action = -1

    all_actions = []
    all_states = []
    player1_values = []
    player2_values = []

    while True:
        try:
            npi = draft.draft_order[draft.next_pick_index]
        except IndexError:
            print(draft.__dict__)
            print(draft.state.__dict__)
            raise IndexError

        if npi < 13:
            if player_1_pick_first:
                action, uct_value, p, nn_value, _ = player1.act(state, action, num_reads=500, deterministic=True)
                player1_values.append(nn_value)
            else:
                action, uct_value, p, nn_value, _ = player2.act(state, action, num_reads=500, deterministic=True)
                player2_values.append(nn_value)
        else:
            if player_1_pick_first:
                action, uct_value, p, nn_value, _ = player2.act(state, action, num_reads=500, deterministic=True)
                player2_values.append(nn_value)
            else:
                action, uct_value, p, nn_value, _ = player1.act(state, action, num_reads=500, deterministic=True)
                player1_values.append(nn_value)

        all_states.append(state.game_state)
        all_actions.append(action)
        state, value, done = draft.step(action)

        if value == 0:  # Dire victory
            print('Dire victory')
            if player_1_pick_first:
                print('New agent won!')
            else:
                print('Old agent won :/')
            break
        elif value == 1:
            print('Radiant Victory')
            if player_1_pick_first:
                print('Old agent won :/')
            else:
                print('New agent won!')
            break
        elif done:
            print('Done but no victory')
            break

    all_actions.append(action)
    all_states.append(state.game_state)

    # TODO: I'm really not confident this is right - it's worth double and triple checking
    all_values = [value] * 23
    player_1_pick_first = [player_1_pick_first] * 23
    del old_model
    del new_model
    empty_cache()
    return dict(all_actions=all_actions, all_states=all_states, all_values=all_values, player1_values=player1_values,
                player2_values=player2_values, player_1_pick_first=player_1_pick_first)


if __name__ == '__main__':
    file_name = 'eval1'
    if file_name is None:
        file_name = f'selfplay_{time.time()}'
    old_model: DraftBert = load('../weights/final_weights/train_from_selfplay_2.torch',
                            map_location=device('cpu'))
    old_model.eval()
    old_model.requires_grad = False

    new_model: DraftBert = load('../data/self_play/memories_for_train_3/new_model.torch',
                            map_location=device('cpu'))
    new_model.eval()
    new_model.requires_grad = False
    memory_size = 500000
    n_jobs = 4
    n_games = 200
    port = 13337
    verbose = True
    hero_ids = pd.read_json('../const/draft_bert_clustering_hero_ids.json', orient='records')

    memory = deque(maxlen=memory_size)
    f = partial(do_rollout, old_model, new_model, hero_ids)

    for batch_of_games in range(n_games // n_jobs):
        start = time.time()
        with Pool(n_jobs) as pool:
            results = pool.map_async(f, [port + i for i in range(n_jobs)]).get()
            memory.extend(results)
        end = time.time()
        print(f'Finished batch {batch_of_games} in {end-start}s')
    with open(f'../data/evaluations/{file_name}.pickle', 'wb') as f:
        pickle.dump(memory, f)

