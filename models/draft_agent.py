import numpy as np
import torch
from models.draft_bert import DraftBert
from collections import deque
import asyncio
import multiprocessing
from draft.draft_env import AllPickState, CaptainModeState
from typing import Union
from solvers.uct import DummyNode, UCTNode
from solvers.enum import Solver
from torch.functional import F

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
    def __init__(self, model: DraftBert, solver, memory_size):
        super().__init__()
        self.model: DraftBert = model
        self.solver = solver
        self.model.masked_output.requires_grad = False
        self.model.matching_output.requires_grad = False
        self.best_model = model
        self.memory = deque(maxlen=memory_size)

    def simulate(self):
        """
        Generates training data of (s, p, v) triplets by playing against itself and storing the results

        :return:
        """

        leaf, value, done, breadcrumbs = self.solver.move_to_leaf()
        value, breadcrumbs = self.evaluate_leaf(leaf, value, done)
        self.solver.backward(leaf, value, breadcrumbs)

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

    def act(self, state, num_reads=1):
        if self.solver is None or state.id not in self.solver.tree:
            self.root = UCTNode(state)
            self.solver = UCT(self.root)
        else:
            self.search.root = self.search.tree[state.id]

        # root = UCTNode(state, move=prior_move, parent=parent)
        for _ in range(num_reads):
            self.simulate()
            # leaf = root.select_leaf()
            # child_priors, value_estimate = (self.model.get_next_hero_output(state),
            #                                 self.model.get_win_output(state))
            # child_priors = child_priors.detach().cpu().numpy()[0]
            # illegal_moves = list(set(range(len(child_priors))) - set(legal_moves))
            # child_priors[illegal_moves] = 0
            # value_estimate = value_estimate.detach().cpu().numpy()[0, 1]
            # leaf.expand(child_priors)
            # leaf.backup(value_estimate)
        self.choose_action()
        return np.argmax(root.child_number_visits), root

    def get_preds(self, state):
        s = state.state
        s_in = torch.LongTensor([s]).unsqueeze(-1)
        mask = torch.zeros_like(s_in)

        encoded_s = self.model.forward(s_in, mask)
        probs = self.model.get_next_hero_output(encoded_s[:, state.draft_order[state.next_pick_index], :])
        probs = probs.detach().cpu().numpy()[0]

        value = self.model.get_win_output(encoded_s[:, 0, :]).detach().cpu().numpy()[0][1]
        legal_moves = state.get_legal_moves
        illegal_moves = np.ones(probs.shape, dtype=bool)
        illegal_moves[legal_moves] = False
        probs[illegal_moves] = -100
        probs = F.softmax(probs, -1)
        return (probs, value, legal_moves)

    def evaluate_leaf(self, leaf, value, done):
        if done == 0:
            probs, value, legal_moves = self.get_preds(leaf.state)
            probs = probs[legal_moves]
            for idx, action in enumerate(legal_moves):
                new_state, _, __ = leaf.state.take_action(action)
                if new_state.id not in self.search.tree:
                    node = UCTNode(new_state)
                    self.search.add_node(node)
                else:
                    node = self.search.tree[new_state.id]
                edge = UCTEdge(leaf, node, )
                leaf.edges.append((action, edge))
        return value

