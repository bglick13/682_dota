from draft.draft_env import AllPickEnv, CaptainModeEnv
from models.draft_agent import DraftAgent
import numpy as np
import pandas as pd
import asyncio
import pickle
import torch
from solvers.enum import Solver


if __name__ == '__main__':
    port = 13337
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')
    draft = CaptainModeEnv(hero_ids, port)
    model = torch.load('../draft_bert_pretrain.torch')
    agent: DraftAgent = DraftAgent(model=model, solver=Solver.UCT, memory_size=100000)
    agent.self_play(draft)
