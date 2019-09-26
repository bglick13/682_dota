import requests
from typing import List, Dict, Union
from secret import API_KEY
from enum import Enum
import pandas as pd


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


class ApiCall:
    def __init__(self, url: str, method: RestMethod = RestMethod.GET, data: Dict = None):
        self.url = url
        self.method = method
        self.data = data
        self._result: Union[ApiCallResponse, None] = None

    def __call__(self, *args, **kwargs):
        if self.result() is not None:
            return self.result()

        if self.method == RestMethod.GET:
            self._result = ApiCallResponse(requests.get(self.url))

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

    def prepare_call(self, url: str, method: RestMethod = RestMethod.GET, data: Dict = None) -> Session:
        url += f'?apy_key={self.api_key}'
        call = ApiCall(url, method, data)
        if call in self.calls:
            del call
            return self
        else:
            self.calls[call] = call
            return self

    def make_call(self) -> ApiCall:
        ret = self._staged()
        self._staged = None
        return ret


def get_matchup(team_a_hero_ids: List[int], team_b_hero_ids: List[int]) -> str:
    """
    Given a list of heroes for each team, return matches with that matchup

    :param team_a_hero_ids: list of integers corresponding to heroes for team a
    :param team_b_hero_ids: list of integers corresponding to heroes for team b
    :return: List[Match]
    """
    base_url = 'https://api.opendota.com/api/findMatches'
    return base_url


def get_hero_ids() -> str:
    base_url = 'https://api.opendota.com/api/heroes'
    return base_url


if __name__ == '__main__':
    sess = Session(API_KEY)

    heros_url = get_hero_ids()
    heros = sess.prepare_call(heros_url).make_call().result().pandas()
    print(heros)