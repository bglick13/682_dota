from abc import ABC
import asyncio
import time
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from grpclib.client import Channel
from dotaservice.protos.DotaService_pb2 import *
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_ALL_DRAFT


class DraftEnv(ABC):
    def __init__(self, heros: pd.DataFrame, port: int):
        self.heros = heros
        self.port = port

        self.SEP = heros.loc[heros['name'] == 'SEP', 'model_id'].values[0]
        self.MASK = heros.loc[heros['name'] == 'MASK', 'model_id'].values[0]
        self.CLS = heros.loc[heros['name'] == 'CLS', 'model_id'].values[0]

        self.draft_order = None

        self.initial_state = self.reset()
        self.next_pick_index = 0
        self._current_state = self.initial_state

        self.TICKS_PER_OBSERVATION = 15
        self.N_DELAY_ENUMS = 5
        self.HOST_TIMESCALE = 10
        self.HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')

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
        return self._current_state[[4, 5, 8, 9, 11]]

    @property
    def dire(self):
        return self._current_state[[16, 17, 20, 21, 23]]

    @property
    def radiant_dota_ids(self):
        return self.heros.loc[self.heros['model_id'].isin(self.radiant), 'id'].values

    @property
    def dire_dota_ids(self):
        return self.heros.loc[self.heros['model_id'].isin(self.dire), 'id'].values

    @property
    def get_legal_moves(self):
        return np.array(list((set(self.heros.loc[~self.heros['name'].isin(['MASK', 'SEP', 'CLS']), 'model_id'].values) -
                              set(self._current_state))))

    def pick(self, hero_id):
        self._current_state[self.draft_order[self.next_pick_index]] = hero_id
        self.next_pick_index += 1

    def reset(self):
        self.next_pick_index = 0
        return np.ones(25) * self.MASK

    def _get_game_config(self):
        radiant_dota_ids = self.heros.loc[self.heros['model_id'].isin(self.radiant), 'id']
        dire_dota_ids = self.heros.loc[self.heros['model_id'].isin(self.dire), 'id']
        hero_picks = []
        for hero in radiant_dota_ids:
            hero_picks.append(HeroPick(team_id=TEAM_RADIANT, hero_id=hero, control_mode=HERO_CONTROL_MODE_DEFAULT))
        for hero in dire_dota_ids:
            hero_picks.append(HeroPick(team_id=TEAM_DIRE, hero_id=hero, control_mode=HERO_CONTROL_MODE_DEFAULT))

        # TODO generate game_id here so it's easily accessible
        return GameConfig(
            ticks_per_observation=self.TICKS_PER_OBSERVATION,
            host_timescale=self.HOST_TIMESCALE,
            host_mode=self.HOST_MODE,
            game_mode=DOTA_GAMEMODE_ALL_DRAFT,
            hero_picks=hero_picks,
        )

    async def play(self, dota_service, config, game_id):

        # Reset and obtain the initial observation. This dictates who we are controlling,
        # this is done before the player definition, because there might be humand playing
        # that take up bot positions.
        config.game_id = game_id
        print(config)
        response = await asyncio.wait_for(dota_service.reset(config), timeout=120)

    async def get_winner(self):
        assert self.done, 'Draft is not complete'
        game_config = self._get_game_config()
        channel_dota = Channel('127.0.0.1', self.port, loop=asyncio.get_event_loop())
        dota_service = DotaServiceStub(channel_dota)

        # game = Game(dota_service=dota_service, max_dota_time=max_dota_time)
        # UUID in game.dota_service.dota_game.game_id
        game_id = str(datetime.datetime.now().strftime('%b%d_%H-%M-%S'))

        config = self._get_game_config()

        try:
            await self.play(dota_service=dota_service, config=config, game_id=game_id)
        except:
            channel_dota.close()
        log_file_path = f'/private/tmp/{game_id}/bots/console.log'
        with open(log_file_path, 'r') as f:
            while True:
                where = f.tell()
                line = f.readline()
                if not line:
                    time.sleep(1)
                    f.seek(where)
                else:
                    if 'Building' in line:
                        print(line)
                    if 'npc_dota_badguys_fort destroyed' in line:
                        print('Radiant Victory')
                        return 1
                    elif 'npc_dota_goodguys_fort destroyed' in line:
                        print('Dire Victory')
                        return 0


class AllPickEnv(DraftEnv):
    def __init__(self, heros: pd.DataFrame, port: int):
        super().__init__(heros, port)
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
        radiant = self.heros.loc[self.heros['model_id'].isin(self.radiant), 'localized_name']
        dire = self.heros.loc[self.heros['model_id'].isin(self.dire), 'localized_name']

        return f'Radiant: {radiant}\nDire: {dire}'


class CaptainModeEnv(DraftEnv):
    def __init__(self, heros: pd.DataFrame, port: int):
        super().__init__(heros, port)
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

