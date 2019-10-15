import torch
from models.draft_bert import DraftBert


class DraftAgent(torch.nn.Module):
    def __init__(self, model: DraftBert, memory_size):
        super().__init__()
        self.model = model
        self.model.masked_output.requires_grad = False
        self.model.matching_output.requires_grad = False
        self.best_model = model
        self.memory_size = memory_size

    def self_play(self, n_games=1):
        """
        Generates training data of (s, p, v) triplets by playing against itself and storing the results

        :return:
        """
        pass

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

    def get_action(self):
        """
        Tree-search to decide an action

        :return:
        """
        pass

