import collections
import numpy as np


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
        self.is_terminal = False
        self.parent = parent
        # self.legal_moves = self.state.get_legal_moves
        self.children = {}
        self.child_priors = np.zeros([len(self.state.heros)], dtype=np.float32)
        self.child_total_value = np.zeros([len(self.state.heros)], dtype=np.float32)
        self.child_number_visits = np.zeros([len(self.state.heros)], dtype=np.float32)

    def is_leaf(self):
        return len(self.edges) == 0

    @property
    def number_visits(self):
        if self.parent is None:
            return 0
        return self.parent.child_number_visits[self.move]
    
    @number_visits.setter
    def number_visits(self, value):
        if self.parent is not None:
            self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        if self.parent is None:
            return 0
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        if self.parent is not None:
            self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self): # 1.25 is the c_puct term that many papers use. It controls exploration.
        return 1.25 * self.child_priors * np.sqrt(np.log(self.number_visits + 1)/(1 + self.child_number_visits))

    def best_child(self):
        if self.state.done:
            return None, None
        values = self.child_Q() + self.child_U()
        legal_moves = self.state.get_legal_moves
        illegal_moves = np.ones(values.shape, dtype=bool)
        illegal_moves[legal_moves] = False
        values[illegal_moves] = -np.inf
        best = np.random.choice(np.flatnonzero(values == values.max()))  # This should randomly draw from tied best
        return best, values[best]

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def add_child(self, move):
        # Value will be 0 unless the game is over
        new_state = self.state.take_action(move)
        self.children[move] = UCTNode(new_state, move = move, parent=self)


class UCT():
    def __init__(self, state, num_rollouts):
        # I'm passing in instantiated nodes instead of just states, is that okay?       -- Yes
        self.root = state
        self.num_rollouts = num_rollouts

    def rollout(self, node=None):
        # TODO: @connor - Does this change work/make sense?
        node = self.root
        while node.is_expanded:
            node.number_visits += 1
            node.total_value -= 1
            move, value = node.best_child()
            if move is None:
                node.is_terminal = True
                break
            if move not in node.children:
                node.add_child(move)
            node = node.children[move]

        # TODO: @connor can I just return the node here and then evaluate and expand from a function in 'agent'?         -- Yes
        return node

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