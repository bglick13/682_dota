import multiprocessing
import numpy as np
from models.draft_bert import DraftBert
from models.draft_agent import DraftAgent
import time
import asyncio
from concurrent.futures import ProcessPoolExecutor
import functools
from draft.draft_env import CaptainModeEnv


# class DraftArena:
#     def __init__(self, agent, env, hero_ids, n_jobs):
#         self.agent: DraftAgent = agent
#         self.envs = [env(hero_ids, 13337 + i) for i in range(n_jobs)]
#         self.n_jobs = n_jobs

async def self_play(agent, hero_ids, n_jobs, n_games):
    envs = [CaptainModeEnv(hero_ids, 13337 + i) for i in range(n_jobs)]
    loop = asyncio.get_event_loop()
    all_states, all_actions, all_values, all_winners = [], [], [], []
    results = []

    for set_of_games in range(n_games//n_jobs):
        executor = ProcessPoolExecutor(max_workers=n_jobs)
        queue = asyncio.Queue()

        jobs = [loop.run_in_executor(executor, agent.self_play, envs[i]) for i in range(n_jobs)]
        for f in asyncio.as_completed(jobs, loop=loop):
            results = await f
            await queue.put(results)
        # for i in range(self.n_jobs):
        #     jobs.append(pool.apply_async(self.agent.self_play, (self.envs[i], ), callback=collect_result))
        #     time.sleep(1)

        # for i, p in enumerate(jobs):
        #     print(f'Getting job {i}')
        #     results.append(p.get())
        #     time.sleep(1)


        for r in results:
            all_states.extend(r[0])
            all_actions.extend(r[1])
            all_values.extend(r[2])
            all_winners.extend(r[3])
    data = np.hstack((all_states, all_actions, all_values))
    agent.memory.extend(data)