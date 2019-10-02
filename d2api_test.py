import d2api
from secret import API_KEY

if __name__ == '__main__':
    api = d2api.APIWrapper(api_key=API_KEY)
    heros = api.get_heroes()
    matches = api.get_match_history(hero_id=1, matches_requested=10000)
    print(matches)