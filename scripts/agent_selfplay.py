#from draft.draft_env import CaptainModeDraft
from models.draft_agent import DraftAgent
import pandas as pd
import numpy as np
import torch
from typing import List
from collections import deque
import pickle
from multiprocessing import Pool
import time


def do_rollout(model, hero_ids, port, verbose=False):
    if not torch.cuda.is_available():
        model = torch.load(model, map_location=torch.device('cpu'))
    else:
        model = torch.load(model)

    players: List[DraftAgent] = [DraftAgent(model=model, pick_first=None),
                                 DraftAgent(model=model, pick_first=None)]
    players = np.random.permutation(players)
    players[0].pick_first = True
    players[1].pick_first = False
    draft = CaptainModeDraft(hero_ids, port)
    state = draft.reset()
    turn = 0
    action = -1

    all_actions = []
    all_states = []

    while True:
        if draft.draft_order[draft.next_pick_index] < 13:
            action, mcts_value, p, nn_value = players[0].act(state, action, num_reads=100)
        else:
            action, mcts_value, p, nn_value = players[1].act(state, action, num_reads=100)
        all_states.append(state)
        all_actions.append(action)
        state, value, done = draft.step(action)
        if verbose:
            print(f'\nTurn {turn}:\nAction: {action}, MCTS Value: {mcts_value}, NN Value: {nn_value}\n{state}')
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
    all_actions.append(action)
    all_states.append(state)

    # TODO: I'm really not confident this is right - it's worth double and triple checking
    all_values = [value] * 22
    # all_values[[0, 2, 4, 6, 9, 11, 13, 15, 17, 19, 20]] = value
    # all_values[[1, 3, 5, 7, 8, 10, 12, 14, 16, 18, 21]] = 1 - value
    return all_actions, all_states, all_values


if __name__ == '__main__':
    memory_size = 500000
    n_jobs = 2
    n_games = 2
    port = 13337
    verbose = True
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')

    memory = deque(maxlen=memory_size)

    start = time.time()
    for batch_of_games in range(n_games // n_jobs):
        # pool = ProcessPoolExecutor(2)
        pool = Pool(n_jobs)
        results = pool.starmap(do_rollout, [('../weights/draft_bert_pretrain.torch', hero_ids, port + i) for i in range(n_jobs)])
        memory.extend(results)

    with open('../data/self_play/memory.pickle', 'wb') as f:
        pickle.dump(memory, f)
    print(f'Played {n_games} games using {n_jobs} jobs in {time.time() - start}s')
