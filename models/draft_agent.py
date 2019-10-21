import numpy as np
import torch
from models.draft_bert import DraftBert
from collections import deque
import asyncio
import multiprocessing
from draft.draft_env import AllPickEnv, CaptainModeEnv
from typing import Union
from solvers.uct import DummyNode, UCTNode
from solvers.enum import Solver


class DummyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = deque(maxlen=1000)

    @staticmethod
    def self_play(env):
        """
        Generates training data of (s, p, v) triplets by playing against itself and storing the results

        :return:
        """
        env.reset()
        states, actions = [], []
        while not env.done:
            s = env.state
            legal_moves = env.get_legal_moves
            a = np.random.choice(legal_moves, 1)

            states.append(s)
            actions.append(a)

            env.pick(a)
        # loop = asyncio.get_event_loop()
        winner = env.get_winner()
        values = np.zeros_like(states)
        if winner == 1:
            values[env.draft_order <= 11] = 1
        else:
            values[env.draft_order > 11] = 1
        return states, actions, values, winner


class DraftAgent(DummyAgent):
    def __init__(self, model: DraftBert, solver: Solver, memory_size):
        super().__init__()
        self.model: DraftBert = model
        self.solver = solver
        self.model.masked_output.requires_grad = False
        self.model.matching_output.requires_grad = False
        self.best_model = model
        self.memory = deque(maxlen=memory_size)

    def self_play(self, env: Union[AllPickEnv, CaptainModeEnv], verbose=0):
        """
        Generates training data of (s, p, v) triplets by playing against itself and storing the results

        :return:
        """
        print(f'starting self-play with env on port: {env.port}')
        env.reset()
        states, actions = [], []
        prior_move, parent = None, None
        turn = 0
        while not env.done:
            if verbose == 1:
                print(f'Turn {turn}\n{env}')
            s = env.state
            s_in = torch.LongTensor([s]).unsqueeze(-1)
            mask = torch.zeros_like(s_in)

            encoded_s = self.model.forward(s_in, mask)
            a, _parent = self.get_action(encoded_s[:, env.next_pick_index, :], prior_move, parent)
            prior_move = a
            parent = _parent

            states.append(s)
            actions.append(a)

            env.pick(a)

        # loop = asyncio.get_event_loop()
        # coro = env.get_winner()
        # winner = loop.run_until_complete(coro)
        # values = np.zeros_like(states)
        # if winner == 1:
        #     values[env.draft_order <= 11] = 1
        # else:
        #     values[env.draft_order > 11] = 1
        # return states, actions, values, winner

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

    def get_action(self, state, prior_move=None, parent=None, num_reads=1):
        if self.solver == Solver.UCT:
            if parent is None:
                parent = DummyNode()
            root = UCTNode(state, move=prior_move, parent=parent)
            for _ in range(num_reads):
                leaf = root.select_leaf()
                child_priors, value_estimate = (self.model.get_next_hero_output(state),
                                                self.model.get_win_output(state))
                child_priors = child_priors.detach().cpu().numpy()[0]
                illegal_moves = list(set(range(len(child_priors))) - set(legal_moves))
                child_priors[illegal_moves] = 0
                value_estimate = value_estimate.detach().cpu().numpy()[0, 1]
                leaf.expand(child_priors)
                leaf.backup(value_estimate)
        return np.argmax(root.child_number_visits), root
