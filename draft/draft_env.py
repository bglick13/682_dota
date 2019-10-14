from abc import ABC
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class Draft(ABC):
    def __init__(self, heros: pd.DataFrame):
        self.heros = heros
        self.SEP = heros.loc[heros['name'] == 'sep', 'model_id']
        self.MASK = heros.loc[heros['name'] == 'mask', 'model_id']
        self.CLS = heros.loc[heros['name'] == 'cls', 'model_id']

        self.draft_order = None

        self.initial_state = self.reset()
        self.next_pick_index = 0
        self._current_state = self.initial_state

    @property
    def done(self):
       return -1

    @property
    def state(self):
        """
        The indices stored here are with respect to the model. They will not necessarily align with the hero_ids DOTA
        uses

        :return:
        """
        return self._current_state

    @property
    def player_turn(self):
        """
        Returns 1 if Dire, 0 if Radiant

        :return:
        """
        return -1

    @property
    def radiant(self):
        return self._current_state[4, 5, 8, 9, 11]

    @property
    def dire(self):
        return self._current_state[16, 17, 20, 21, 23]

    def get_legal_moves(self):
        return set(self.heros['model_id'].values) - set(self._current_state)

    def pick(self, hero_id):
        self._current_state[self.draft_order[self.next_pick_index]] = hero_id
        self.next_pick_index += 1

    def reset(self):
        self.next_pick_index = 0
        return np.ones(25) * self.MASK


class AllPick(Draft):
    def __init__(self, heros: pd.DataFrame):
        super().__init__(heros)
        self.draft_order = [4,
                            16, 17,
                            5, 8,
                            20, 21,
                            9, 11,
                            23]

    @property
    def done(self):
        return self.next_pick_index >= 10

    @property
    def player_turn(self):
        return self.next_pick_index >= 5

    @property
    def state(self):
        """
        Override to mask out the ban indices

        :return:
        """
        return np.concatenate([self.CLS],  # Always start with the CLS token - index [0]

                              # Radiant - starts at index 1
                              np.ones(3) * self.MASK,  # First wave of 3 bans - indices [1, 2, 3]
                              self._current_state[:2],  # First two picks - indices [4, 5]
                              np.ones(2) * self.MASK,  # Two more bans - indices [6, 7]
                              self._current_state[2:4],  # Two more picks - indices [8, 9]
                              np.ones(1) * self.MASK,  # Final ban - index [10]
                              self._current_state[4:5],  # Final pick - index [11]
                              [self.SEP],  # Index [12]

                              # Dire - starts at index 13
                              np.ones(3) * self.MASK,  # First wave of 3 bans - indices [13, 14, 15]
                              self._current_state[5:7],  # First two picks - indices [16, 17]
                              np.ones(2) * self.MASK,  # Two more bans - indices [18, 19]
                              self._current_state[7:9],  # Two more picks - indices [20, 21]
                              np.ones(1) * self.MASK,  # Final ban - index [22]
                              self._current_state[9:],  # Final pick - index [23]
                              [self.SEP]  # Index [24]
                              )

    def __str__(self):
        if self.heros:
            radiant = self.heros.loc[self.heros['model_id'].isin(self.radiant), 'localized_name']
            dire = self.heros.loc[self.heros['model_id'].isin(self.dire), 'localized_name']

            return f'Radiant: {radiant}\nDire: {dire}'


class CaptainMode(Draft):
    def __init__(self, heros: pd.DataFrame):
        super().__init__(heros)
        self.draft_order = [0, 11, 1, 12, 2, 13,
                            3, 14, 4, 15,
                            5, 16, 6, 17,
                            7, 18, 8, 19,
                            9, 20,
                            10, 21]

    @property
    def player_turn(self):
        """
        Returns 1 if Dire, 0 if Radiant

        :return:
        """

        return self.next_pick_index >= 13

    @property
    def done(self):
        return self.next_pick_index >= 24

