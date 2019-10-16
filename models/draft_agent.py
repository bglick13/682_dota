import numpy as np
import torch
from models.draft_bert import DraftBert
from collections import deque
import asyncio
import multiprocessing
from draft.draft_env import AllPickEnv, CaptainModeEnv
from typing import Union


class DummyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = deque(maxlen=1000)

    @staticmethod
    async def self_play(env: Union[AllPickEnv, CaptainModeEnv]):
        """
        Generates training data of (s, p, v) triplets by playing against itself and storing the results

        :return:
        """
        print(f'starting self-play with env on port: {env.port}')

        env.reset()
        states, actions = [], []
        while not env.done:
            s = env.state
            legal_moves = env.get_legal_moves
            a = np.random.choice(legal_moves, 1)

            states.append(s)
            actions.append(a)

            env.pick(a)
        loop = asyncio.get_event_loop()
        coro = env.get_winner()
        winner = await coro()
        values = np.zeros_like(states)
        if winner == 1:
            values[env.draft_order <= 11] = 1
        else:
            values[env.draft_order > 11] = 1
        return states, actions, values, winner


class DraftAgent(DummyAgent):
    def __init__(self, model: DraftBert, memory_size):
        super().__init__()
        self.model = model
        self.model.masked_output.requires_grad = False
        self.model.matching_output.requires_grad = False
        self.best_model = model
        self.memory = deque(maxlen=memory_size)

    def self_play(self, env: Union[AllPickEnv, CaptainModeEnv]):
        """
        Generates training data of (s, p, v) triplets by playing against itself and storing the results

        :return:
        """
        print(f'starting self-play with env on port: {env.port}')
        env.reset()
        states, actions = [], []
        while not env.done:
            s = env.state
            s_in = torch.LongTensor([s]).unsqueeze(-1)
            mask = torch.zeros_like(s_in)

            encoded_s = self.model.forward(s_in, mask)
            a = self.get_action(encoded_s[:, env.next_pick_index, :])

            states.append(s)
            actions.append(a)

            env.pick(a)
        loop = asyncio.get_event_loop()
        coro = env.get_winner()
        winner = loop.run_until_complete(coro)
        values = np.zeros_like(states)
        if winner == 1:
            values[env.draft_order <= 11] = 1
        else:
            values[env.draft_order > 11] = 1
        return states, actions, values, winner

    def update_network(self, batch_size, steps):
        """
        Train self.model for n steps with samples from the self-play dataset.
        Loss is a combination of:
            - cross_entropy(predicted_policy, actual_policy)
            - mean_squared_error(predicted_value, actual_outcome)

        :return:
        """
        pass

    def evaluate_network(self):
        """
        Latest neural network and current best neural network play a series of games. Winner is selected as current
        best.

        :return:
        """
        pass

    def get_action(self, s):
        """
        Tree-search to decide an action

        :return:
        """
        pass
