import numpy as np
from torch import nn, FloatTensor, LongTensor
from models.draft_bert import DraftBert
from collections import deque
from typing import Union
from draft.draft_env import DraftState
from search.uct2 import UCTNode, UCT
from torch.functional import F


class DummyAgent(nn.Module):
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
        winner = env.get_winner()
        values = np.zeros_like(states)
        if winner == 1:
            values[env.draft_order <= 11] = 1
        else:
            values[env.draft_order > 11] = 1
        return states, actions, values, winner


class DraftAgent(DummyAgent):
    def __init__(self, model: DraftBert, pick_first):
        super().__init__()
        self.model: DraftBert = model
        # if torch.cuda.is_available():
        #     self.model.cuda()
        self.solver = None
        self.model.masked_output.requires_grad = False
        self.model.matching_output.requires_grad = False
        self.best_model = model
        self.action_size = model.n_heros
        self.pick_first = pick_first

    def simulate(self):
        """

        :return:
        """

        leaf = self.solver.rollout()
        value = self.evaluate_leaf(leaf)
        self.solver.backup(leaf, value)
        return leaf

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

    def act(self, state, action=-1, num_reads=100):
        if self.solver is None:
            self.root = UCTNode(state, action)
            # self.root.number_visits += 1
            self.solver = UCT(self.root, num_reads)
        else:
            self.root = UCTNode(state, action, self.root)
            # self.root.number_visits += 1
            self.solver.root = self.root

        leafs = []
        for _ in range(num_reads):
            leafs.append(self.simulate())
        action, value, values = self.root.best_child()
        next_state = state.take_action(action)
        nn_value = self.get_preds(next_state)[1]
        p = F.softmax(FloatTensor(values), -1).numpy()
        return action, values, p, nn_value, leafs

    def get_preds(self, leaf: Union[UCTNode, DraftState]):
        if isinstance(leaf, UCTNode):
            state = leaf.state
        else:
            state = leaf
        s = state.state
        s[0] = self.model.cls
        s[12] = self.model.sep
        s[-1] = self.model.sep
        s_in = LongTensor([s])
        s_in.requires_grad = False
        # if torch.cuda.is_available():
        #     s_in = s_in.cuda()

        try:
            encoded_s = self.model.forward(s_in)
        except Exception as e:
            print(e)
            print(s_in)
        if state.next_pick_index < 22:
            probs = self.model.get_next_hero_output(encoded_s[:, state.draft_order[state.next_pick_index], :])
            probs = probs[0]
            legal_moves = state.get_legal_moves
            illegal_moves = np.ones(probs.shape, dtype=bool)
            illegal_moves[legal_moves] = False
            probs[illegal_moves] = -100
            probs = F.softmax(probs, -1).detach().cpu().numpy()
        else:
            probs = None
            legal_moves = None

        # This should be all it takes to make sure the network is always player-centric
        value = F.softmax(self.model.get_win_output(encoded_s[:, 0, :]), -1).detach().cpu().numpy(
            )[0][int(self.pick_first)]
        return probs, value, legal_moves

    def evaluate_leaf(self, leaf):
        probs, value, legal_moves = self.get_preds(leaf)
        if not leaf.is_terminal:
            leaf.expand(probs)
        return value

    # def choose_action(self):
    #     action, value = self.root.best_child()
    #     return action, value
