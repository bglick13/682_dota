import collections
import random
import math

from IPython import embed

import numpy as np
import time

# import draft.draft_env
# import models.draft_agent
# import models.draft_bert

class UCTNode(object):
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move
        self.is_expanded = False
        self.parent = parent
        self.legal_moves = random.shuffle(self.state.getLegalMoves())
        self.children = {}
        self.child_priors = np.zeros([len(legal_moves)], dtype=np.float32)
        self.child_total_value = np.zeros([len(legal_moves)], dtype=np.float32)
        self.child_number_visits = np.zeros([len(legal_moves)], dtype=np.float32)

    def is_leaf(self):
        return len(self.edges) == 0

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]
    
    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return nn.evaluate(self.state)[1] * np.sqrt(np.log(self.number_visits)/(1 + self.child_number_visits))

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def add_child(self, move):
        self.children[move] = UCTNode(self.state.play(move), move = move, parent=self)


class UCT():
    def __init__(self, state, num_rollouts):
        root = UCTNode(state, move = None, parent = DummyNode())
        self.num_rollouts = num_rollouts

    def rollout(self, node):
        while node.is_expanded:
            node.number_visits += 1
            node.total_value -= 1
            move = node.best_child()
            if move not in self.children:
                self.add_child(move)
            node = self.children[move]

        child_priors, value_estimate = nn.evaluate(node.state)
        node.expand(child_priors)
        return value_estimate

    def backup(self, node, value_estimate):
        while node.parent is not None:
            node.number_visits += 1
            node.total_value += value_estimate #*self.state.to_play
            node = node.parent


def UCT_search (state, num_reads):
    root = UCTNode(state, move = None, parent = DummyNode())
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = nn.evaluate(leaf.state)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    return np.argmax(root.child_number_visits)