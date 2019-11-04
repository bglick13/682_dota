import os
import time
from abc import ABC

import numpy as np
import pandas as pd

try:
    from dotaservice.protos.DotaService_pb2 import *
    from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_ALL_DRAFT
    dotaservice = True
except ModuleNotFoundError:
    dotaservice = False
    print('dotaservice not found')
import uuid
import pickle
import docker
import copy
import logging
import re

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


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
            logger.debug(e)
            logger.debug(f'Radiant dota ids : {radiant_dota_ids}\Radiant model ids: {self.radiant}')
            logger.debug(f'Dire dota ids : {dire_dota_ids}\Dire model ids: {self.dire}')

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
        local_volume = f'../rollout_results'
        local_volume = os.path.dirname(os.path.abspath(local_volume))
        local_volume = os.path.join(local_volume, 'rollout_results')
        # if not os.path.isdir(local_volume):
            # os.mkdir(local_volume)
        local_volume = os.path.join(local_volume, game_id)
        os.mkdir(local_volume)
        config.game_id = game_id
        with open(f'{local_volume}/config.pickle', 'wb') as f:
            pickle.dump(config, f)
        client = docker.from_env()
        assert os.path.isdir(local_volume), 'Incorrect mount point'
        job_number = self.port - 13337
        cpus = f'{job_number*2}-{job_number*2+1}'
        print(f'Job number {job_number} working on cpus {cpus}')
        container = client.containers.run('dotaservice',
                                          volumes={local_volume: {'bind': '/tmp', 'mode': 'rw'}},
                                          # ports={f'{self.port}/tcp': self.port},
                                          cpuset_cpus=cpus,
                                          cpu_period=50000,
                                          cpu_quota=49900,
                                          remove=True,
                                          detach=True)
        logger.debug('launched container')
        client.close()
        return container

    def get_winner(self):
        assert self.done, 'Draft is not complete'
        game_id = str(uuid.uuid1())
        cutoff_1 = 1 * 60
        cutoff_2 = 2 * 60
        cutoff_3 = 3 * 60
        cutoff_4 = 4 * 60
        cutoff_5 = 5 * 60
        cutoff_6 = 6 * 60
        config = self._get_game_config()
        client = docker.from_env()
        container = self._play(config=config, game_id=game_id)
        local_volume = f'../rollout_results'
        local_volume = os.path.dirname(os.path.abspath(local_volume))
        local_volume = os.path.join(local_volume, 'rollout_results')
        local_volume = os.path.join(local_volume, game_id)
        log_file_path = f'{local_volume}/{game_id}/bots/console.log'
        start = time.time()
        time.sleep(30)
        radiant_progress = 0
        dire_progress = 0

        win_check_1 = None
        win_check_2 = None
        win_check_3 = None
        win_check_4 = None
        win_check_5 = None
        win_check_6 = None
        # while container in client.containers.list():
        #     pass
        try:
            with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                f.write("\nStarting new game.")
            with open(log_file_path, 'r') as f:
                # buffer = f.read()
                # if 'goodguys_fort destroyed' in buffer:
                #     return 0
                # elif 'badguys_fort destroyed' in buffer:
                #     return 1
                print(f'Opened file: {log_file_path}')
                while True:
                    tmp_time = time.time()
                    if tmp_time - start > cutoff_1 and win_check_1 is None:
                        win_check_1 = (radiant_progress > dire_progress)
                    if tmp_time - start > cutoff_2 and win_check_2 is None:
                        win_check_2 = (radiant_progress > dire_progress)
                    if tmp_time - start > cutoff_3 and win_check_3 is None:
                        win_check_3 = (radiant_progress > dire_progress)
                    if tmp_time - start > cutoff_4 and win_check_4 is None:
                        win_check_4 = (radiant_progress > dire_progress)
                    if tmp_time - start > cutoff_5 and win_check_5 is None:
                        win_check_5 = (radiant_progress > dire_progress)
                    if tmp_time - start > cutoff_6 and win_check_6 is None:
                        win_check_6 = (radiant_progress > dire_progress)
                    where = f.tell()
                    line = f.readline()
                    if not line:
                        time.sleep(10)
                        f.seek(where)
                    else:
                        if 'Building' in line:
                            print(f'{game_id} : {line}')
                            if 'badguys' in line:
                                radiant_progress += 1
                            if 'goodguys' in line:
                                dire_progress += 1

                            if 'npc_dota_badguys_fort destroyed' in line:
                                print(f'{game_id} : Radiant Victory')
                                if win_check_1 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("1 min check is " + str(win_check_1 == True))
                                if win_check_2 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("2 min check is " + str(win_check_2 == True))
                                if win_check_3 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("3 min check is " + str(win_check_3 == True))
                                if win_check_4 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("4 min check is " + str(win_check_4 == True))
                                if win_check_5 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("5 min check is " + str(win_check_5 == True))
                                if win_check_6 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("6 min check is " + str(win_check_6 == True))

                                return 1
                            elif 'npc_dota_goodguys_fort destroyed' in line:
                                print(f'{game_id} : Dire Victory')
                                if win_check_1 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("1 min check is " + str(win_check_1 == False))
                                if win_check_2 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("2 min check is " + str(win_check_2 == False))
                                if win_check_3 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("3 min check is " + str(win_check_3 == False))
                                if win_check_4 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("4 min check is " + str(win_check_4 == False))
                                if win_check_5 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("5 min check is " + str(win_check_5 == False))
                                if win_check_6 is not None:
                                    with open('tmp_log_file_for_win_check_test.txt', 'a+') as f:
                                        f.write("6 min check is " + str(win_check_6 == False))

                                return 0
        except FileNotFoundError as e:
            logger.error(e)

        client.close()

    def __str__(self):
        radiant = self.heros.loc[self.heros['model_id'].isin(self.radiant), 'localized_name']
        radiant_bans = self.heros.loc[self.heros['model_id'].isin(self.radiant_bans), 'localized_name']
        dire = self.heros.loc[self.heros['model_id'].isin(self.dire), 'localized_name']
        dire_bans = self.heros.loc[self.heros['model_id'].isin(self.dire_bans), 'localized_name']
        out = f'Radiant Bans:\n{radiant_bans}\nDire Bans\n{dire_bans}'
        out += f'\nRadiant\n{radiant}\nDire\n{dire}'

        return out
