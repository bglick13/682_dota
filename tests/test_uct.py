from draft.draft_env import CaptainModeDraft
from models.draft_agent import DraftAgent
import pandas as pd

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

    radiant_player: DraftAgent = DraftAgent(model=model, memory_size=100000, pick_first=True)
    dire_player: DraftAgent = DraftAgent(model=model, memory_size=100000, pick_first=False)
    state = draft.reset()
    turn = 0
    action = -1

    while True:
        if draft.draft_order[draft.next_pick_index] < 13:
            action, nn_value = radiant_player.act(state, action, num_reads=100)
        else:
            action, nn_value = dire_player.act(state, action, num_reads=100)
        state, value, done = draft.step(action)
        print(f'\nTurn {turn}:\nAction: {action}, Value: {nn_value}\n{state}')
        if value == -1:  # Dire victory
            print('Dire victory')
            break
        elif value == 1:
            print('Radiant Victory')
            break
        turn += 1

