import os, sys

sys.path.append(os.path.join('..'))

from draft.draft_env import CaptainModeDraft
from models.draft_agent import DraftAgent, DraftBert
import pandas as pd
import numpy as np
import torch
from typing import List
from collections import deque
import pickle
from multiprocessing import Pool
import time


def do_rollout(model, hero_ids, port, verbose=False):
    # if not torch.cuda.is_available():
    # model: DraftBert = torch.load(model, map_location=torch.device('cpu'))
    # model.eval()
    # # else:
    # #     model = torch.load(model)
    # model.requires_grad = False

    player = DraftAgent(model=model, pick_first=np.random.choice([True, False]))
    draft = CaptainModeDraft(hero_ids, port)
    state = draft.reset()
    turn = 0
    action = -1

    all_actions = []
    all_states = []

    while True:
        try:
            npi = draft.draft_order[draft.next_pick_index]
        except IndexError:
            print(draft.__dict__)
            print(draft.state.__dict__)
            raise IndexError

        if npi < 13:
            if player.pick_first:
                action, mcts_value, p, nn_value = player.act(state, action, num_reads=100)
            else:
                legal_moves = draft.state.get_legal_moves
                action = np.random.choice(legal_moves)
        else:
            if player.pick_first:
                legal_moves = draft.state.get_legal_moves
                action = np.random.choice(legal_moves)
            else:
                action, mcts_value, p, nn_value = player.act(state, action, num_reads=100)
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
        print("Agent victory!")
    else:
        print('Agent Lost :(')
    all_actions.append(action)
    all_states.append(state.game_state)

    # TODO: I'm really not confident this is right - it's worth double and triple checking
    all_values = [value] * 22
    all_agent_pick_first = [player.pick_first] * 22
    # all_values[[0, 2, 4, 6, 9, 11, 13, 15, 17, 19, 20]] = value
    # all_values[[1, 3, 5, 7, 8, 10, 12, 14, 16, 18, 21]] = 1 - value
    del model
    torch.cuda.empty_cache()
    return dict(all_actions=all_actions, all_states=all_states, all_values=all_values,
                all_agent_pick_first=all_agent_pick_first)


if __name__ == '__main__':
    model = torch.load('../weights/final_weights/draft_bert_pretrain_all_pick.torch',
                                  map_location=torch.device('cpu'))
    model.eval()
    model.requires_grad = False

    memory_size = 500000
    n_jobs = 4
    n_games = 4
    port = 13337
    verbose = True
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')

    memory = deque(maxlen=memory_size)

    start = time.time()
    for batch_of_games in range(n_games // n_jobs):
        # pool = ProcessPoolExecutor(2)
        with Pool(n_jobs) as pool:
        # pool = Pool(n_jobs)
            results = pool.starmap(do_rollout, [(model, hero_ids, port + i) for i in range(n_jobs)])
            memory.extend(results)
    end = time.time()

    with open('../data/self_play/allpick_vs_random_memory.pickle', 'wb') as f:
        pickle.dump(memory, f)

    print(f'Played {n_games} games using {n_jobs} jobs in {time.time() - start}s')
