from draft.draft_env import CaptainModeDraft, DraftState
from models.draft_agent import DraftAgent
import numpy as np
import pandas as pd
import asyncio
import pickle
import torch


if __name__ == '__main__':
    port = 13337
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')
    print('creating env')
    draft = CaptainModeDraft(hero_ids)
    print('env created')
    print('loading model')
    model = torch.load('../draft_bert_pretrain.torch', map_location=torch.device('cpu'))
    print('model loaded')

    radiant_player: DraftAgent = DraftAgent(model=model, memory_size=100000)
    dire_player: DraftAgent = DraftAgent(model=model, memory_size=100000)
    state = draft.reset()
    turn = 0
    action = -1
    try:
        while True:
            if draft.draft_order[draft.next_pick_index] < 13:
                action, value = radiant_player.act(state, action)
            else:
                action, value = dire_player.act(state, action)
            state, value, done = draft.step(action)
            print(f'\nTurn {turn}:\n{state}')
            if value == -1:  # Dire victory
                print('Dire victory')
                break
            elif value == 1:
                print('Radiant Victory')
                break
            turn += 1
    except:
        print('Final game state:\n')
        print(state)

