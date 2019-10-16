from draft.arena import self_play
from models.draft_agent import DummyAgent
from draft.draft_env import CaptainModeEnv
import pandas as pd
import asyncio


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')
    agent = DummyAgent()
    self_play(agent, hero_ids, 2, 2)
    print(agent.memory)