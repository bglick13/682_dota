from draft.draft_env import AllPickEnv
import numpy as np
import pandas as pd
import asyncio


if __name__ == '__main__':
    port = 13337
    hero_ids = pd.read_json('../const/hero_ids.json', orient='records')
    draft = AllPickEnv(hero_ids, port)
    while not draft.done:
        moves = draft.get_legal_moves
        move = np.random.choice(moves, 1)
        draft.pick(move)
    print(draft)
    loop = asyncio.get_event_loop()
    coro = draft.get_winner()
    winner = loop.run_until_complete(coro)