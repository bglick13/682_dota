import requests
from typing import List, Dict, Union, Tuple
from secret import API_KEY, STEAM_KEY
from enum import Enum
import numpy as np
import pandas as pd
import time
from pandas.io.json import json_normalize

class RestMethod(Enum):
    GET = 1
    POST = 2


class ApiCallResponse:
    def __init__(self, response: requests.Response):
        self.response = response

    def json(self):
        return self.response.json()

    def pandas(self):
        return pd.DataFrame(self.json())


class MatchApiCallResponse(ApiCallResponse):
    def json(self):
        return self.response.json()['result']['matches']

    def pandas(self):
        try:
            json = self.json()
            cm_json = [j for j in json if j['game_mode'] == 2]
            cm_json = [j for j in cm_json if 'picks_bans' in j]
            cm = json_normalize(cm_json, ['picks_bans'], ['radiant_win', 'start_time', 'match_id', 'duration',
                                                                'game_mode', 'match_seq_num'])
        except:
            cm = []

        try:
            ap_json = [j for j in json if j['game_mode'] == 22]

            ap = json_normalize(ap_json, ['players'], ['radiant_win', 'start_time', 'match_id', 'duration',
                                                                'game_mode', 'match_seq_num'])
            return ap, cm
        except:
            ap = []
        return ap, cm

class ApiCall:
    def __init__(self, url: str, method: RestMethod = RestMethod.GET, call_type: ApiCallResponse = None, data: Dict = None):
        if call_type is None:
            self.call_type = ApiCallResponse
        else:
            self.call_type = call_type
        self.url = url
        self.method = method
        self.data = data
        self._result: Union[call_type, None] = None

    def __call__(self, *args, **kwargs):
        if self.result() is not None:
            return self.result()

        if self.method == RestMethod.GET:
            success = False
            while not success:
                ret = requests.get(self.url, params=self.data)
                if ret.status_code < 400:
                    success = True
                else:
                    time.sleep(30)
                # if ret.
            self._result = self.call_type(ret)

        # Currently unsupported and probably unnecessary
        elif self.method == RestMethod.POST:
            self._result = ApiCallResponse(requests.post(self.url, self.data))
        return self

    def result(self):
        return self._result


class Session:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls: Dict[ApiCall, ApiCall] = dict()
        self._staged: Union[ApiCall, None] = None

    def _prepare_call(self, url: str, method: RestMethod = RestMethod.GET, data: Dict = None, call_type=ApiCallResponse):
        call = ApiCall(url, method=method, data=data, call_type=call_type)
        if call in self.calls:
            del call
        else:
            self.calls[call] = call
        self._staged = call
        return self

    def get_matches_by_seq_number(self, start_id) -> ApiCall:
        base_url = r'https://api.steampowered.com/IDOTA2Match_570/GetMatchHistoryBySequenceNum/v0001/'
        params = dict(key=self.api_key, start_at_match_seq_num=start_id, matches_requested=100)
        matches = self._prepare_call(base_url, data=params, call_type=MatchApiCallResponse).make_call()
        return matches

    def make_call(self) -> ApiCall:
        ret = self._staged()
        self._staged = None
        return ret


def get_matchup(team_a_hero_ids: List[int], team_b_hero_ids: List[int]) -> Tuple[str, Union[dict, None]]:
    """
    Given a list of heroes for each team, return matches with that matchup

    :param team_a_hero_ids: list of integers corresponding to heroes for team a
    :param team_b_hero_ids: list of integers corresponding to heroes for team b
    :return: List[Match]
    """
    joint_hero_ids = np.append(team_a_hero_ids, team_b_hero_ids)
    hero_id_query = '%2C'.join(joint_hero_ids.astype(str))
    base_url = 'https://api.opendota.com/api/findMatches'
    params = dict(teamA=team_a_hero_ids, teamB=team_b_hero_ids)
    # base_url = (f'https://api.opendota.com/api/explorer?sql=select%20public_matches.match_id%2C%20public_matches.'
    #             f'radiant_win%2C%20public_matches.start_time%2C%0Aarray_agg(player_matches.player_slot)%20as%20'
    #             f'player_slots%2C%0Aarray_agg(player_matches.hero_id)%20h%0Afrom%20public_matches%0Ajoin%20matches%20'
    #             f'using%20(match_id)%0AJOIN%20player_matches%20using(match_id)%0Awhere%20player_matches.hero_id%20in%20'
    #             f'({hero_id_query})%0Agroup%20by%20public_matches.match_id%20having%20count(*)%20%3D%20{len(joint_hero_ids)}')

    return base_url, None


def get_matches_by_sequence_number(start_id: int, n_pages: int=1):
    base_url = r'https://api.steampowered.com/IDOTA2Match_570/GetMatchHistoryBySequenceNum/v0001/'
    params = dict(key=STEAM_KEY, start_at_match_seq_num=start_id)
    return base_url, params


def get_hero_ids() -> Tuple[str, Union[dict, None]]:
    base_url = 'https://api.opendota.com/api/heroes'
    return base_url, None


if __name__ == '__main__':
    sess = Session(API_KEY)

    matches = sess.get_matches_by_seq_number(501088752)
    df = matches.result().pandas()

    # heros_url, _ = get_hero_ids()
    # heros = sess.prepare_call(heros_url).make_call().result().pandas()
    #
    # matchup_url, _ = get_matchup([7, 81, 40, 98], [18, 106, 35, 101, 62])
    # print(matchup_url)
    # matchup = sess.prepare_call(matchup_url).make_call()

    print(heros)