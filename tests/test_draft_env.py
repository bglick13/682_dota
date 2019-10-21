from draft.draft_env import AllPickState, CaptainModeState
import numpy as np
import pandas as pd
import asyncio
import pickle


if __name__ == '__main__':
    port = 13337
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')
    draft = CaptainModeState(hero_ids, port)
    while not draft.done:
        moves = draft.get_legal_moves
        move = np.random.choice(moves, 1)

        draft.pick(move)
    print(draft)
    # with open('config.pickle', 'wb') as f:
    #     pickle.dump(draft._get_game_config(), f)
    # loop = asyncio.get_event_loop()
    # coro = draft.get_winner()
    # winner = loop.run_until_complete(coro)