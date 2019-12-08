import sys,os
sys.path.append('..')

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


def do_rollout(model_1, model_2, hero_ids, port, verbose=False):
    # player_1_pick_first = np.random.choice([True, False])
    player_1_pick_first = True
    player1: DraftAgent = DraftAgent(model=model_1, pick_first=player_1_pick_first, greedy=False)
    player2: DraftAgent = DraftAgent(model=model_2, pick_first=not player_1_pick_first , greedy=True)
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
                action, uct_value, nn_value = player1.act(state, action, num_reads=500, eps=0, deterministic=True)
                player1_values.append(nn_value)
            else:
                action, uct_value, nn_value = player2.act(state, action, num_reads=500, eps=0, deterministic=True)
                player2_values.append(nn_value)
        else:
            if player_1_pick_first:
                action, uct_value, nn_value = player2.act(state, action, num_reads=500, eps=0, deterministic=True)
                player2_values.append(nn_value)
            else:
                action, uct_value, nn_value = player1.act(state, action, num_reads=500, eps=0, deterministic=True)
                player1_values.append(nn_value)

        all_states.append(state.game_state)
        all_actions.append(action)
        state, value, done = draft.step(action)

        with open('eval_results.txt', mode='a+') as f:
            if value == 0:  # Dire victory
                print('Dire victory')
                if player_1_pick_first:
                    print('New agent won!')
                    f.write('New agent\n')
                else:
                    print('Old agent won :/')
                    f.write('Old agent\n')
                break
            elif value == 1:
                print('Radiant Victory')
                if player_1_pick_first:
                    print('Old agent won :/')
                    f.write('Old agent\n')
                else:
                    print('New agent won!')
                    f.write('New agent\n')
                break
            elif done:
                print('Done but no victory')
                f.write('No winner')
                break

    all_actions.append(action)
    all_states.append(state.game_state)

    # TODO: I'm really not confident this is right - it's worth double and triple checking
    all_values = [value] * 23
    player_1_pick_first = [player_1_pick_first] * 23
    del model_1
    del model_2
    empty_cache()
    return dict(all_actions=all_actions, all_states=all_states, all_values=all_values, player1_values=player1_values,
                player2_values=player2_values, player_1_pick_first=player_1_pick_first)


if __name__ == '__main__':
    file_name = 'uct_vs_greedy_eval_0_2'
    if file_name is None:
        file_name = f'selfplay_{time.time()}'

    model_1: DraftBert = load('../weights/final_weights/draft_bert_pretrain_captains_mode_with_clusters.torch',
                            map_location=device('cpu'))
    model_1.eval()
    model_1.requires_grad = False

    model_2: DraftBert = load('../weights/final_weights/draft_bert_pretrain_captains_mode_with_clusters.torch',
                            map_location=device('cpu'))
    model_2.eval()
    model_2.requires_grad = False
    memory_size = 500000
    n_jobs = 4  
    n_games = 100
    port = 13337
    verbose = True
    hero_ids = pd.read_json('../const/draft_bert_clustering_hero_ids.json', orient='records')

    memory = deque(maxlen=memory_size)
    f = partial(do_rollout, model_1, model_2, hero_ids)

    for batch_of_games in range(n_games // n_jobs):
        start = time.time()
        with Pool(n_jobs) as pool:
            results = pool.map_async(f, [port + i for i in range(n_jobs)]).get()
            memory.extend(results)
        with open(f'../data/evaluations/{file_name}.pickle', 'wb') as file:
            pickle.dump(memory, file)
        end = time.time()
        print(f'Finished batch {batch_of_games} in {end-start}s')
    

