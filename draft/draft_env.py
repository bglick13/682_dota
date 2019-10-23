import os
from abc import ABC
import asyncio
import time
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
try:
    from dotaservice.protos.DotaService_pb2 import *
    from dotaservice.protos.DotaService_grpc import DotaServiceStub
    from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_ALL_DRAFT
except ModuleNotFoundError:
    dotaservice = False
    print('dotaservice not found')
import uuid
import pickle
import docker
import copy


class CaptainModeDraft:
    def __init__(self, heros: pd.DataFrame, port: int):
        self.draft_order = np.array([1, 13, 2, 14, 3, 15,
                                     4, 16, 5, 17,
                                     6, 18, 7, 19,
                                     8, 20, 9, 21,
                                     10, 22,
                                     11, 23])
        self.next_pick_index = 0
        self.heros = heros
        self.port = port
        self.SEP = heros.loc[heros['name'] == 'SEP', 'model_id'].values[0]
        self.MASK = heros.loc[heros['name'] == 'MASK', 'model_id'].values[0]
        self.CLS = heros.loc[heros['name'] == 'CLS', 'model_id'].values[0]
        self.state = DraftState(np.ones(25) * self.MASK, self.next_pick_index, self.port, heros)

    def reset(self):
        self.next_pick_index = 0
        return self.state

    def step(self, action):
        next_state = self.state.take_action(action)
        self.next_pick_index += 1
        self.state = next_state

        value = -1
        done = 0

        if self.state.done:
            print('Final State:\n')
            print(self.state)
            value = self.state.get_winner()
            done = 1

        return next_state, value, done


class DraftState(ABC):
    def __init__(self, state, next_pick_index, port, heros):
        self.port = port
        self.game_state = state
        self.next_pick_index = next_pick_index

        self.heros = heros
        self.SEP = heros.loc[heros['name'] == 'SEP', 'model_id'].values[0]
        self.MASK = heros.loc[heros['name'] == 'MASK', 'model_id'].values[0]
        self.CLS = heros.loc[heros['name'] == 'CLS', 'model_id'].values[0]

        self.draft_order = np.array([1, 13, 2, 14, 3, 15,
                                     4, 16, 5, 17,
                                     6, 18, 7, 19,
                                     8, 20, 9, 21,
                                     10, 22,
                                     11, 23])

        self.next_pick_index = next_pick_index
        if dotaservice:
            self.TICKS_PER_OBSERVATION = 15
            self.N_DELAY_ENUMS = 5
            self.HOST_TIMESCALE = 10
            self.HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')

    @property
    def id(self):
        return f'{self.game_state}'

    @property
    def playerTurn(self):
        """
        Returns 1 if Dire, 0 if Radiant

        :return:
        """

        return self.next_pick_index >= 13

    @property
    def done(self):
        return self.next_pick_index > 21

    @property
    def state(self):
        """
        The indices stored here are with respect to the model. They will not necessarily align with the hero_ids DOTA
        uses

        :return:
        """
        return self.game_state

    @property
    def radiant(self):
        return self.game_state[[4, 5, 8, 9, 11]]

    @property
    def radiant_bans(self):
        return self.game_state[[1, 2, 3, 6, 7, 10]]

    @property
    def dire(self):
        return self.game_state[[16, 17, 20, 21, 23]]

    @property
    def dire_bans(self):
        return self.game_state[[13, 14, 15, 18, 19, 22]]

    @property
    def radiant_dota_ids(self):
        return self.heros.loc[self.heros['model_id'].isin(self.radiant), 'id'].values

    @property
    def dire_dota_ids(self):
        return self.heros.loc[self.heros['model_id'].isin(self.dire), 'id'].values

    @property
    def get_legal_moves(self):
        return np.array(list((set(self.heros.loc[~self.heros['name'].isin(['MASK', 'SEP', 'CLS']), 'model_id'].values) -
                              set(self.game_state))))

    def pick(self, hero_id):
        self.game_state[self.draft_order[self.next_pick_index]] = hero_id

        self.next_pick_index += 1

    def take_action(self, action):
        new_state = copy.deepcopy(self.state)
        new_state[self.draft_order[self.next_pick_index]] = action
        new_state = DraftState(new_state, self.next_pick_index+1, self.port, self.heros)

        return new_state

    def _get_game_config(self):
        radiant_dota_ids = self.heros.loc[self.heros['model_id'].isin(self.radiant), 'id'].values
        dire_dota_ids = self.heros.loc[self.heros['model_id'].isin(self.dire), 'id'].values
        hero_picks = []
        try:
            for hero in radiant_dota_ids:
                hero_picks.append(HeroPick(team_id=TEAM_RADIANT, hero_id=hero, control_mode=HERO_CONTROL_MODE_DEFAULT))
            for hero in dire_dota_ids:
                hero_picks.append(HeroPick(team_id=TEAM_DIRE, hero_id=hero, control_mode=HERO_CONTROL_MODE_DEFAULT))
        except ValueError as e:
            print(e)
            print(radiant_dota_ids)
            print(dire_dota_ids)

        # TODO generate game_id here so it's easily accessible
        return GameConfig(
            ticks_per_observation=self.TICKS_PER_OBSERVATION,
            host_timescale=self.HOST_TIMESCALE,
            host_mode=self.HOST_MODE,
            game_mode=DOTA_GAMEMODE_ALL_DRAFT,
            hero_picks=hero_picks,
        )

    def _play(self, config, game_id):

        # Reset and obtain the initial observation. This dictates who we are controlling,
        # this is done before the player definition, because there might be humand playing
        # that take up bot positions.
        os.mkdir(f'../{game_id}')
        config.game_id = game_id
        with open(f'../{game_id}/config.pickle', 'wb') as f:
            pickle.dump(config, f)
        print('calling reset')
        client = docker.from_env()
        container = client.containers.run('dotaservice',
                              volumes={f'/Users/benjaminglickenhaus/PycharmProjects/682_project/{game_id}': {'bind':'/tmp',
                                                                                                             'mode': 'rw'}},
                              ports={f'{self.port}/tcp': self.port},
                              detach=True)
        print('launched container')
        time.sleep(10)
        print(container.logs())

    def get_winner(self):
        assert self.done, 'Draft is not complete'
        game_id = str(uuid.uuid1())

        config = self._get_game_config()
        print(f"Calling play for game id: {game_id}")

        self._play(config=config, game_id=game_id)
        log_file_path = f'../{game_id}/{game_id}/bots/console.log'
        time.sleep(10)
        with open(log_file_path, 'r') as f:
            print(f'Opened file: {log_file_path}')
            while True:
                where = f.tell()
                line = f.readline()
                if not line:
                    time.sleep(1)
                    f.seek(where)
                else:
                    if 'Building' in line:
                        print(f'{line}')
                    if 'npc_dota_badguys_fort destroyed' in line:
                        print(f'Radiant Victory')
                        return 1
                    elif 'npc_dota_goodguys_fort destroyed' in line:
                        print(f'Dire Victory')
                        return 0

    def __str__(self):
        radiant = self.heros.loc[self.heros['model_id'].isin(self.radiant), 'localized_name']
        radiant_bans = self.heros.loc[self.heros['model_id'].isin(self.radiant_bans), 'localized_name']
        dire = self.heros.loc[self.heros['model_id'].isin(self.dire), 'localized_name']
        dire_bans = self.heros.loc[self.heros['model_id'].isin(self.dire_bans), 'localized_name']
        out = f'Radiant Bans:\n{radiant_bans}\nDire Bans\n{dire_bans}'
        out += f'\nRadiant\n{radiant}\nDire\n{dire}'

        return out
