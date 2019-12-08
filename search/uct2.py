import collections
import numpy as np

class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


class UCTNode(object):
    def __init__(self, state, move, parent=None, running_avg=False):
        self.state = state
        self.move = move
        self.is_expanded = False
        self.is_terminal = False
        self.parent = parent
        self.number_visits = 0
        self.total_value = 0.0
        self.children = {}
        self.child_priors = np.zeros([len(self.state.heros)], dtype=np.float32)
        self.child_total_value = np.zeros([len(self.state.heros)], dtype=np.float32)
        self.child_number_visits = np.zeros([len(self.state.heros)], dtype=np.float32)
        self.q_running_mean = None
        self.u_running_mean = None
        self.running_avg = running_avg

    def is_leaf(self):
        return len(self.edges) == 0

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        # return 5 * self.child_priors * np.sqrt(np.log(self.number_visits + 1)/(1 + self.child_number_visits))
        if self.running_avg:
            return np.sqrt(np.log(self.number_visits + 1)/(1 + self.child_number_visits))
        else:
            return 5 * self.child_priors * np.sqrt(np.log(self.number_visits + 1) / (1 + self.child_number_visits))

    def best_child(self):
        if self.state.done:
            return None, None, None
        q = self.child_Q()
        u = self.child_U()
        if self.running_avg:
            if self.q_running_mean is None:
                self.q_running_mean = q
                self.u_running_mean = u
            else:
                self.q_running_mean = (.9 * self.q_running_mean) + (.1 * q)
                self.u_running_mean = (.9 * self.u_running_mean) + (.1 * u)
            q = (q - self.q_running_mean) / (self.q_running_mean + 1e-5)
            u = (u - self.u_running_mean) / (self.u_running_mean + 1e-5)
        values = q + u

        legal_moves = self.state.get_legal_moves
        illegal_moves = np.ones(values.shape, dtype=bool)
        illegal_moves[legal_moves] = False
        values[illegal_moves] = -np.inf
        best = np.random.choice(np.flatnonzero(np.isclose(values, values.max())))  # This should randomly draw from tied best
        # if np.random.uniform() <= self.eps:
        #     np.random.seed()
        #     best = np.random.choice(np.flatnonzero([values != -np.inf]))
        return best, values[best], values

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def add_child(self, move):
        # Value will be 0 unless the game is over
        new_state = self.state.take_action(move)
        self.children[move] = UCTNode(new_state, move = move, parent=self, running_avg=self.running_avg)


class UCT():
    def __init__(self, state, num_rollouts):
        self.root = state
        self.num_rollouts = num_rollouts

    def rollout(self, node=None):
        node = self.root
        # depth = 0
        while node.is_expanded:
            node.number_visits += 1
            if node.parent is not None:
                node.parent.child_number_visits[node.move] += 1
            move, value, values = node.best_child()
            # print(f'Depth: {depth}, Q: {node.child_Q()[move]}, U: {node.child_U()[move]}')
            if move is None:
                node.is_terminal = True
                break
            if move not in node.children:
                node.add_child(move)
            node = node.children[move]
            # depth += 1

        return node

    def backup(self, node, value_estimate):
        node.number_visits += 1
        node.total_value += value_estimate
        if node.parent is not None:
            node.parent.child_number_visits[node.move] += 1
            node.parent.child_total_value[node.move] += value_estimate
            node = node.parent
        while node.parent is not None:
            node.total_value += value_estimate
            node.parent.child_total_value[node.move] += value_estimate
            node = node.parent


def UCT_search (state, num_rollouts):
    root = UCTNode(state, move = None, parent = DummyNode())
    for _ in range(num_rollouts):
        leaf = root.select_leaf()
        child_priors, value_estimate = nn.evaluate(leaf.state)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    return np.argmax(root.child_number_visits)