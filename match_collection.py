from api_calls import Session
from secret import STEAM_KEY
import pickle
from tqdm import tqdm
import pandas as pd
import networkx as nx
import datetime
import time
import numpy as np


def insert_matchup_to_graph(radiant, dire, win):
    try:
        matchup_results = matchups[radiant][dire]
        matchup_results['wins'].append(win)
        matchup_results['match_ids'].append(key)
    except KeyError:
        matchups.add_edge(radiant, dire)
        matchup_results = matchups[radiant_heros][dire]
        matchup_results['wins'] = [win]
        matchup_results['match_ids'] = [key]


if __name__ == '__main__':

    sess = Session(STEAM_KEY)
    # 1000 pages is ~ 1 day of DOTA
    pages = 1000

    # 501088752 corresponds to a match 2014-03-05
    # 4233710726 corresponds to 2019-9-29
    # 3223910726 corresponds to 2018-2-1
    start_id = 3223910726
    sequence_length = 100000000
    old_id = start_id
    save_frequency = 1000
    i = 0

    graph_checkpoint = 'tmp/test_matchups_3126814594.pkl'
    df_checkpoint = 'test_draft_order_df_3126814594.pkl'
    if graph_checkpoint is not None:
        with open(graph_checkpoint, 'rb') as f:
            matchups = pickle.load(f)
    else:
        matchups = nx.Graph()

    if df_checkpoint is not None:
        with open(df_checkpoint, 'rb') as f:
            pick_order_df = pickle.load(f)
        start_id = pick_order_df['match_seq_num'].max()
    else:
        pick_order_df = None

    seq_numbers = np.arange(start_id-sequence_length, start_id, 132)
    for p in tqdm(seq_numbers):
        start = time.time()
        matches = sess.get_matches_by_seq_number(start_id)
        ap, cm = matches.result().pandas()

        # Ranked Captain's mode games only
        old_id = start_id
        if len(ap) == 0 and len(cm) == 0:
            print(f'No results returned for sequence number: {start_id}')
            start_id += 1
            time.sleep(1.1)
            continue
        elif len(cm) == 0 and len(ap) > 0:
            start_id = ap['match_seq_num'].max()
        else:
            start_id = max(ap['match_seq_num'].max(), cm['match_seq_num'].max())

        # Parse the all-pick games
        if len(ap) > 0:
            for key, grp in ap.groupby('match_id'):
                radiant_heros = tuple(sorted(grp.head()['hero_id'].values))
                dire_heroes = tuple(sorted(grp.tail()['hero_id'].values))

                if radiant_heros not in matchups:
                    matchups.add_node(radiant_heros)
                if dire_heroes not in matchups:
                    matchups.add_node(dire_heroes)

                winning_team = radiant_heros if grp['radiant_win'].values[0] else dire_heroes
                insert_matchup_to_graph(radiant_heros, dire_heroes, winning_team)

        if len(cm) > 0:
            # Parse the captain's mode games
            for key, grp in cm.groupby('match_id'):
                picked_heros = grp.loc[grp['is_pick'], :]
                radiant_heros = tuple(sorted(picked_heros.loc[picked_heros['team'] == 0, 'hero_id'].values))
                dire_heros = tuple(sorted(picked_heros.loc[picked_heros['team'] == 1, 'hero_id'].values))
                winning_team = radiant_heros if grp['radiant_win'].values[0] else dire_heroes
                insert_matchup_to_graph(radiant_heros, dire_heroes, winning_team)

            if pick_order_df is None:
                pick_order_df = cm
            else:
                pick_order_df = pd.concat((pick_order_df, cm))

        duration = time.time() - start
        time.sleep(max(1.1-duration, .1))
        if (i + 1) % save_frequency == 0:
            with open(f'tmp/test_matchups_{p}.pkl', 'wb') as f:
                pickle.dump(matchups, f)
            with open(f'test_draft_order_df_{p}.pkl', 'wb') as f:
                pickle.dump(pick_order_df, f)
        i += 1

    pick_order_df['start_time'] = pick_order_df['start_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    with open('test_matchups.pkl', 'wb') as f:
        pickle.dump(matchups, f)
    with open('test_draft_order_df.pkl', 'wb') as f:
        pickle.dump(pick_order_df, f)
