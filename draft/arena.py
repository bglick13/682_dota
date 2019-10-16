import multiprocessing
import numpy as np
from models.draft_bert import DraftBert
from models.draft_agent import DraftAgent
import time
import asyncio
from concurrent.futures import ProcessPoolExecutor
import functools
from draft.draft_env import CaptainModeEnv
from aiomultiprocess import Pool
import docker


client = docker.from_env()

# class DraftArena:
#     def __init__(self, agent, env, hero_ids, n_jobs):
#         self.agent: DraftAgent = agent
#         self.envs = [env(hero_ids, 13337 + i) for i in range(n_jobs)]
#         self.n_jobs = n_jobs

async def self_play(agent, hero_ids, n_jobs, n_games):
    envs = [CaptainModeEnv(hero_ids, 13337 + i) for i in range(n_jobs)]
    all_states, all_actions, all_values, all_winners = [], [], [], []
    results = []

    for set_of_games in range(n_games//n_jobs):

        async with Pool(2) as pool:
            result = await pool.map(agent.self_play, envs)

        for r in results:
            all_states.extend(r[0])
            all_actions.extend(r[1])
            all_values.extend(r[2])
            all_winners.extend(r[3])
    data = np.hstack((all_states, all_actions, all_values))
    agent.memory.extend(data)