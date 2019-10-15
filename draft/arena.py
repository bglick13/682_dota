import multiprocessing
import numpy as np
from models.draft_bert import DraftBert
from models.draft_agent import DraftAgent
import time


class DraftArena:
    def __init__(self, agent, env, hero_ids, n_jobs):
        self.agent: DraftAgent = agent
        self.envs = [env(hero_ids, 13337 + i) for i in range(n_jobs)]
        self.n_jobs = n_jobs

    def self_play(self, n_games):
        def collect_result(result):
            results.append(result)

        all_states, all_actions, all_values, all_winners = [], [], [], []
        for set_of_games in range(n_games//self.n_jobs):
            pool = multiprocessing.Pool(self.n_jobs)
            results = []
            p = []
            for i in range(self.n_jobs):
                p.append(pool.apply_async(self.agent.self_play, (self.envs[i], ), callback=collect_result))
            for job in p:
                job.get()
                time.sleep(1)
            pool.close()
            pool.join()
            for r in results:
                all_states.extend(r[0])
                all_actions.extend(r[1])
                all_values.extend(r[2])
                all_winners.extend(r[3])
        data = np.hstack((all_states, all_actions, all_values))
        self.agent.memory.extend(data)