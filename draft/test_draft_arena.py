from draft.arena import DraftArena
from models.draft_agent import DummyAgent
from draft.draft_env import CaptainModeEnv
import pandas as pd


if __name__ == '__main__':
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')
    agent = DummyAgent()
    arena = DraftArena(agent=agent, env=CaptainModeEnv, hero_ids=hero_ids, n_jobs=2)
    arena.self_play(2)
    print(agent.memory)