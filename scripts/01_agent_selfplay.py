import os, sys
sys.path.append('..')

import pickle
import time
import cProfile
from collections import deque
from functools import partial
from multiprocessing import Pool

import pandas as pd
from torch import load, device
from torch.cuda import empty_cache

from draft.draft_env import CaptainModeDraft
from models.draft_agent import DraftAgent, DraftBert


def do_rollout(model, hero_ids, num_reads, port, verbose=False, eps=0.1):
    # player_1_pick_first = np.random.choice([True, False])
    player1: DraftAgent = DraftAgent(model=model, pick_first=True)
    player2: DraftAgent = DraftAgent(model=model, pick_first=False)
    draft = CaptainModeDraft(hero_ids, port)
    state = draft.reset()
    turn = 0
    action = -1

    # TODO: record and train on final cluster values here
    all_actions = []
    all_states = []
    player1_nn_values = []
    player2_nn_values = []
    player1_uct_values = []
    player2_uct_values = []

    while True:
        try:
            npi = draft.draft_order[draft.next_pick_index]
        except IndexError:
            print(draft.__dict__)
            print(draft.state.__dict__)
            raise IndexError

        if npi < 13:
            action, uct_value, p, nn_value, leafs = player1.act(state, action, num_reads=num_reads, eps=eps)
            if len(all_actions) == 6:
                action = model.le.transform([129])[0]
            player1_nn_values.append(nn_value)
            player1_uct_values.append(uct_value)
            # player1_uct_rollout_leafs.append(leafs)
        else:
            action, uct_value, p, nn_value, leafs = player2.act(state, action, num_reads=num_reads, eps=eps)
            player2_nn_values.append(nn_value)
            player2_uct_values.append(uct_value)
            # player2_uct_rollout_leafs.append(leafs)

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

    all_actions.append(action)
    all_states.append(state.game_state)

    # TODO: I'm really not confident this is right - it's worth double and triple checking
    all_values = [value] * 23
    del model
    empty_cache()
    return dict(all_actions=all_actions, all_states=all_states, all_values=all_values, player1_nn_values=player1_nn_values,
                player2_nn_values=player2_nn_values, player1_uct_values=player1_uct_values,
                player2_uct_values=player2_uct_values,)


if __name__ == '__main__':
    file_name = None
    if file_name is None:
        file_name = f'selfplay_{time.time()}'
    model: DraftBert = load('../weights/final_weights/train_from_selfplay_1.torch',
                            map_location=device('cpu'))
    model.eval()
    model.requires_grad = False
    memory_size = 500000
    n_jobs = 1
    n_games = 1
    port = 13337
    verbose = True
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')

    memory = deque(maxlen=memory_size)
    # do_rollout(model, hero_ids, port)
    f = partial(do_rollout, model, hero_ids, 500)

    for batch_of_games in range(n_games // n_jobs):
        start = time.time()

        start_batch = time.time()
        with Pool(n_jobs) as pool:
            results = pool.map_async(f, [port + i for i in range(n_jobs)]).get()
            memory.extend(results)
        with open(f'../data/self_play/{file_name}.pickle', 'wb') as file:
            pickle.dump(memory, file)
        end = time.time()
        print(f'Finished batch {batch_of_games} in {end-start}s')
