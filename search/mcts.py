import math
import collections

import numpy as np

def DummyNode(object):
    """ Fake dummy node to be placeholder parent of root node. """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

class MCTSNode(object):
    """ MCTS search tree node.

        Node computes the action scores of all of its children and makes
        a decision about which mvoee to explore next. Update the children
        dictionary after new node is selected.

        Params:
            - position: dictionary of
                    draft: 2 x 5 array of ints representing heroes, -1 is unset
                    bans: 2 x 3 array of ints representing heroes, -1 is unset
            - p_move: a prior move (selection of hero) that led to this position (int)
            - parent: a parent MCTSNode
    """

    def __init__(self, position, p_move = None, parent = None):
        if parent is None:
            parent = DummyNode()
        self.position = position
        self.p_move = p_move
        self.parent = parent
        self.is_expanded = False
        self.losses_applied = 0

        self.illegal_moves = 1 - self.position.all_legal_moves()
        self.child_N = np.zeros([], dtype=np.float32)
        self.child_W = np.zeros([], dtype=np.float32)

        self.original_prior = np.zeros([], dtype=np.float32)
        self.child_prior = np.zeros([], dtype=np.float32)
        self.children = {}

    def __repr__(self):
        return "<MCTSNode move=%s, N=%s, to_play=%s>" % (
            self.position.recent[-1:], self.N, self.position.to_play)

    @property
    def child_action_score(self):
        return  (self.child_Q * self.position.to_play +
                 self.child_U - 1000 * self.illegal_moves)

    @property
    def child_Q(self):
        return self._child_W / (1 + self.child_N)

    @property
    def chil_U(self):
        return ((2.0 * (math.log(
                (1.0 + self.N + FLAGS.c_puct_base) / FLAGS.c_puct_base)
                     + FLAGS.c_puct_init)) * math.sqrt(max(1, self.N - 1)) * 
                 self.child_prior / (1 + self.child_N))

    @property
    def Q(self):
        return self.W / (1 + self.N)

    @property
    def N(self):
        return self.parent.child_N[self.p_move]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.p_move] = value

    @property
    def W(self):
        return self.parent.child_W[self.p_move]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.p_move] = value

    @property
    def Q_perspective(self):
        return self.Q * self.position.to_play

    def select_leaf(self):
        current = self
        pass_move = go.N * go.N
        while True:
            if not current.is_expanded:
                break
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, fcoord):
        if fcoord not in self.children:
            new_position = self.position.play_move(
                coords.from_flat(fcoord))
            self.children[fcoord] = MCTSNode(
                new_position, p_move = fcoord, parent = self)
        return self.children[fcoord]

    def add_virtual_loss(self, up_to):
        """ Propagate virtual loss up to the root node.

        Params:
            up_to: The node to proagate until
        """
        self.losses_applied += 1
        loss = self.position.to_play
        self.W += loss
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def incorporate_results(self, move_probabilities, value, up_to):
        assert move_probabilities.shape == (N * N + 1)
        assert not self.position.is_game_over()

        if self.is_expanded:
            return
        self.is_expanded = True

        move_probs = move_probabilities * (1 - self.illegal_moves)
        scale = sum(move_probs)
        if scale > 0:
            move_probs *= 1 / scale

        self.original_prior = self.child_prior = move_probs

        self.child_W = np.ones([], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """ Propagates a value estimation up to the root node.

        Params:
            - value: the value to be propagated
            - up_to: the node to propagate until.
        """
        self.N += 1
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self):
        """ True if the position is at a move greater than max depth
            or if draft and ban are full.
        """
        return self.position.is_game_over() or self.position.n >= FLAGS.max_game_length

    def inject_noise(self):
        epsilon = 1e-15
        legal_moves = (1 - self.illegal_moves) + epsilon
        a = legal_moves * ([FLAGS.dirichlet_noise_alpha] * (N * N + 1))
        dirichlet = np.random.dirichlet(a)
        self.child_prior = (self.child_prior * (1 - FLAGS.dirichlet_noise_weight) + 
                            dirichlet * FLAGS.dirichlet_noise_weight)

    def children_as_pi(self, squash=Fallse):
        """ Return the child visit counts as a probability distribution, pi
            If squash is true, exponentiate the probabilities by a temperature
            slightly larger than unity to encourage diversity in early play and
            hopefully to move away from 3-3s
        """
        probs = self.child_N
        if squash:
            probs = probs ** .98
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            return porbs
        return probs / np.sum(probs)

    def best_child(self):
        return np.argmax(self.child_N + self.child_action_score / 10000)

    def most_visited_path_nodes(self):
        node = self
        output = []
        while node.children:
            node = node.children.get(node.best_child())
            assert node is not None
            output.append(node)
        return output

    def most_visited_path(self):
        output = []
        node = self
        for node in self.most_visited_path_nodes():
            # Need to add to this.
            pass
        output.append("Q: {.5f}\n".format(node.Q))
        return ''.join(output)

    def mvp_gg(self):
        """ Return most visited path. """
        output = []
        for node in self.most_visited_path_nodes():
            if max(node.child_N) <= 1:
                break
            # do something here now
        return ' '.join(output)

    def rank_children(self):
        ranked_children = list(range(N * N + 1))
        ranked_children.sort(key=lambda i: (
            self.child_N[i], self.child_action_score[i]), reverse=True)
        return ranked_children

    def describe(self):
        ranked_children = self.rank_children()
        soft_n = self.child_N / max(1, sum(self.child_N))
        prior = self.child_prior
        p_delta = soft_n - prior
        p_rel = np.divide(p_delta, prior, out=np.zeros_like(
            p_delta), where=prior != 0)
        # Stats Dump
        output = []
        output.append("")

        for i in ranked_children[:15]:
            if self.child_N[i] == 0:
                break
            output.append("")

        return ''.join(output)