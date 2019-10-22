import collections
import random
import math


import numpy as np
import time
import copy

# import draft.draft_env
# import models.draft_agent
# import models.draft_bert


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


class UCTNode(object):
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move
        self.is_expanded = False
        self.parent = parent
        # self.legal_moves = random.shuffle(self.state.getLegalMove())
        self.children = {}
        self.child_priors = np.zeros([136], dtype=np.float32)
        self.child_total_value = np.zeros([136], dtype=np.float32)
        self.child_number_visits = np.zeros([136], dtype=np.float32)

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

    def select_leaf(self):
        current = self
        while current.is_expanded:
            current.number_visits += 1
            current.total_value -= 1
            move = current.best_child()
            if move not in self.children:
                self.add_child(move)
            current = self.children[move]
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def add_child(self, move):
        self.children[move] = UCTNode(self.state.play(move), move = move, parent=self)

    def backup(self, value_estimate):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            # TODO: Connor is unconvinced this should be commented out - he's wrong
            current.total_value += value_estimate  #* self.state.to_play
            current = current.parent


def UCT_search (state, num_reads):
    root = UCTNode(state, move = None, parent = DummyNode())
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = nn.evaluate(leaf.state)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    return np.argmax(root.child_number_visits)


class nn():
    @classmethod
    def evaluate(self, state):
        return np.random.random([136]), np.random.random()


class State():
    def __init__(self, to_play=1):
        self.to_play = to_play

    def play(self, move):
        return State(-self.to_play)


# num_reads = 1000
# tick = time.time()
# UCT_search(State(), num_reads)
# tock = time.time()
# print("Took %s sec to run %s times" % (tock - tick, num_reads))
# # import resource
# # print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)