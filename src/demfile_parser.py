import os

from data_csgo_tournament.csgo_tournament_metadata import *
from data_csgo_tournament.csgo_weapons import *

from functools import lru_cache
import json

import pandas as pd
import numpy as np
import ast


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# %% WEAPONS
def get_round_weapons_fired(df):
    weapons = set()
    for p in df[df['event'] == 'weapon_fire']['parameters']:
        weapons.add(ast.literal_eval(p)['weapon'])
    return weapons


def is_round_pistol(df, weapons=None):
    if weapons is None:
        weapons = get_round_weapons_fired(df)
    return len(weapons.intersection(set(SMGS + RIFLES + HEAVY))) == 0 and \
        len(weapons.intersection(set(PISTOLS))) > 0


# %% PLAYERS
def get_match_players(df):
    users = set()
    for e in df[df['event'] == 'player info']['parameters']:
        item = ast.literal_eval(e)
        if item['fakeplayer'] == '0':
            users.add(item['name'])
    return users


# %% EVENTS
def get_footstep_players(df):
    users = set()
    for e in df[df['event'] == 'player_footstep']['parameters']:
        item = ast.literal_eval(e)
        user_name = item['userid'].split(' (id:')[0]
        users.add(user_name)
    return users


# %% ROUNDS
def get_round_initial_positions(df, round_users):
    positions_T = []
    positions_CT = []
    users = round_users.copy()
    for e in df[df['event'] == 'player_footstep']['parameters']:
        item = ast.literal_eval(e)
        user_found = None
        for u in users:
            if u == item['userid'][:len(u)]:
                if 'userid position' in item:
                    pos_coordinates = ast.literal_eval(item['userid position'])
                    if 'userid team' in item:
                        if item['userid team'] == 'T':
                            positions_T.append(pos_coordinates)
                            user_found = u
                            break
                        if item['userid team'] == 'CT':
                            positions_CT.append(pos_coordinates)
                            user_found = u
                            break
        if user_found is not None:
            users.remove(user_found)
        if users == []:
            break
    return positions_T, positions_CT


def is_position_initial(pos, team_type):
    x, y, z = pos
    if team_type == 'T':
        # if not (x < 1400 and 100 < y < 900):
        #     print(team_type, x, y)
        return x < -1400 and 100 < y < 900
    elif team_type == 'CT':
        # if not (2200 < x and 1600 < y < 2500):
        #     print(team_type, x, y)
        return 2200 < x and 1600 < y < 2500
    else:
        return None


def is_players_initial(positions_T, positions_CT):
    if len(positions_T) == 0 or len(positions_CT) == 0:
        return False

    p = [is_position_initial(pos, 'T') for pos in positions_T]
    p += [is_position_initial(pos, 'CT') for pos in positions_CT]

    return np.all(p)


def get_round_stat(df, tickrate):
    rounds = []

    for round_info, round_df in split_match_into_rounds(df, tickrate):
        if 'round_freeze_end' in round_df['event'].values:
            local_begin = round_df[round_df['event'] == 'round_freeze_end'].index[0]
        else:
            local_begin = round_df.index[0]

        if 'round_end' in round_df['event'].values:
            local_end = round_df[round_df['event'] == 'round_end'].index[0]
        else:
            local_end = round_df.index[-1]

        round_local = round_df.loc[local_begin:local_end]
        weapons = get_round_weapons_fired(round_local)
        round_info['weapons'] = weapons
        round_info['is_pistol'] = is_round_pistol(round_local, weapons)

        rounds.append((round_df, round_info))

    return rounds


def get_monotonic_tail(v):
    tail = v[-1:]
    elem_pred = tail[-1] if len(tail) > 0 else None
    for elem in v[-2::-1]:
        if elem < elem_pred:
            tail = [elem] + tail
            elem_pred = elem
        else:
            break
    return tail


# %% MATCH
def split_match_into_rounds(df, tickrate):
    df.loc[df.shape[0]]=[df.iloc[-1].tick+1,'round_officially_ended','{}']

    start_i = 0

    rounds = []
    key_events = ['round_prestart', 'round_start', 'round_freeze_end', 'round_end']
    key_events_orders = {e: i for i, e in enumerate(key_events)}

    for i, off_end_i in enumerate(df[df['event'] == 'round_officially_ended'].index):
        df_tmp = df.loc[start_i:off_end_i, :]

        if (i==(len(df[df['event']=='round_officially_ended'].index)-1)):
            ind_to_skip = df_tmp[df_tmp.event=='round_freeze_end'].index[1]
            df_tmp = df_tmp.query("index != @ind_to_skip")

        key_idx = df_tmp[df_tmp['event'].isin(key_events)].index
        round_tail_orders = get_monotonic_tail([key_events_orders[e] for e in df_tmp.loc[key_idx, 'event']])

        round_info = {'round_control_points': df_tmp.loc[key_idx, 'event'].values}

        if 'weapon_fire' in df_tmp['event'].values and 'player_footstep' in df_tmp['event'].values:
            if key_events_orders['round_freeze_end'] in round_tail_orders:
                tail_len = len(round_tail_orders)
                round_df = df_tmp.loc[key_idx[-tail_len]:]

                effective_time = (round_df['tick'].iloc[-1] - round_df[round_df['event'] == 'round_freeze_end']['tick'].iloc[0]) / tickrate
                effective_time_check = effective_time > MIN_ROUND_DURATION
                round_info['effective_time'] = effective_time

                time_limit_check = True
                if key_events_orders['round_start'] in round_tail_orders:
                    item = ast.literal_eval(round_df[round_df['event'] == 'round_start']['parameters'].iloc[0])
                    time_limit_check = item['timelimit'] == '115'

                footstep_users = get_footstep_players(round_df)
                number_of_users_check = len(footstep_users) == NUMBER_OF_PLAYERS
                round_info['round_footstep_players'] = sorted(footstep_users)

                if effective_time_check and time_limit_check and number_of_users_check:
                    rounds.append((round_info, round_df))
            else:
                footstep_users = get_footstep_players(df_tmp)
                pos_T, pos_CT = get_round_initial_positions(df_tmp, footstep_users)
                if is_players_initial(pos_T, pos_CT):
                    footsteps = df_tmp[df_tmp['event'] == 'player_footstep']
                    round_df = df_tmp.loc[footsteps.index[0]:]

                    effective_time = (round_df['tick'].iloc[-1] - round_df['tick'].iloc[0]) / tickrate
                    effective_time_check = effective_time > MIN_ROUND_DURATION
                    round_info['effective_time'] = effective_time

                    number_of_users_check = len(footstep_users) == NUMBER_OF_PLAYERS
                    round_info['round_footstep_players'] = sorted(footstep_users)

                    if effective_time_check and number_of_users_check:
                        rounds.append((round_info, round_df))

        start_i = off_end_i
    return rounds


@lru_cache(None)
def get_match_info(n_match, path):
    path_to_csv = os.path.join(path, parsed_demo_filename[n_match-1])
    print(f'fetching info on match {n_match} : {path_to_csv}')

    df = pd.read_csv(path_to_csv)
    tickrate = 128  # default
    rounds_list = get_round_stat(df, tickrate)

    return rounds_list


@lru_cache(None)
def get_game_context(path):
    game_context = {}
    for n_match, _ in MATCH_LEN.items():
        rounds_list = get_match_info(n_match, path)
        rounds = {}
        players = get_players(n_match)
        for n_round, round_ in enumerate(rounds_list):
            round_data = round_[0].copy()
            round_data['users_self'] = round_data.parameters.apply(ast.literal_eval).apply(lambda x: x.get('userid'))
            round_data['users_self'] = round_data['users_self'].apply(
                lambda x: players.index(x.split()[0]) if x and x.split()[0] in players else None)
            round_data['users_attacker'] = round_data.parameters.apply(ast.literal_eval).apply(lambda x: x.get('attacker'))
            round_data['users_attacker'] = round_data['users_attacker'].apply(
                lambda x: players.index(x.split()[0]) if x and x.split()[0] in players else None)

            round_data['ms'] = (((round_data['tick'] - round_data.tick.iloc[0]) / 128) * 1000)
            rounds[n_round] = round_data
        # dmps = json.dumps(rounds, indent=4, cls=NpEncoder)
        # print(dmps, file=open(f"prepared_{parsed_demo_filename[n_match-1][:-len('.csv')]}.json", "wt"))
        game_context[n_match] = rounds
    return game_context


# %% DATA STRUCTURE
@lru_cache(None)
def get_context_vector(n_player, n_match, n_round, start_time, path):
    game_context = get_game_context(path)
    context_vector = np.zeros(12)
    bomb_interaction_events = ['bomb_pickup', 'bomb_beginplant', 'bomb_planted', 'bomb_exploded', 'bomb_begindefuse',
                               'bomb_defused', 'bomb_abortplant']
    if n_round >= 2:
        end_time = game_context[n_match][n_round - 2].iloc[-1].ms
        if start_time == 0 and n_round - 3 >= 0:
            prev_round = game_context[n_match][n_round - 3]
            start_ = prev_round.iloc[-1].ms - 5000
            df = prev_round.query('ms>=@start_ and ms<=@start_+3000')
        elif start_time == 3000 and n_round - 3 >= 0:
            prev_round = game_context[n_match][n_round - 3]
            start_ = game_context[n_match][n_round - 3].iloc[-1].ms - 2000
            df_1 = prev_round.query('ms>=@start_')
            df_2 = game_context[n_match][n_round - 2].query('ms<=1000')
            df = df_1.append(df_2, ignore_index=True)
        elif start_time == 3000 and n_round - 3 < 0:
            df = game_context[n_match][n_round - 2].query('ms<=1000')
        elif start_time - 5000 >= end_time:
            df = game_context[n_match][n_round - 2].query('ms>=@end_time-6000')
        else:
            df = game_context[n_match][n_round - 2].query('ms>=@start_time-5000 and ms<=@start_time-2000')

        # check events
        # self player
        if df.query('event=="player_hurt" and  users_self == @n_player').values.size:
            context_vector[0] = 1
        if df.query('event=="player_death" and  users_self == @n_player').values.size:
            context_vector[1] = 1
        if df.query('event=="player_blind" and  users_self == @n_player').values.size:
            context_vector[2] = 1
        if df.query('event in @bomb_interaction_events and  users_self == @n_player').values.size:
            context_vector[3] = 1

        # teammate player
        if df.query(
                'event=="player_hurt" and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[4] = 1
        if df.query(
                'event=="player_death" and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[5] = 1
        if df.query(
                'event in @bomb_interaction_events and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[6] = 1

        # enemy player killed/hurted by player
        if df.query(
                'event=="player_hurt" and  users_self != @n_player and users_attacker == @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[7] = 1
        if df.query(
                'event=="player_death" and  users_self != @n_player and users_attacker == @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[8] = 1

        # enemy player killed/hurted by another player
        if df.query(
                'event=="player_hurt" and  users_self != @n_player and users_attacker != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[9] = 1
        if df.query(
                'event=="player_death" and  users_self != @n_player and users_attacker != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[10] = 1

        # enemy player interacted with bomb
        if df.query(
                'event in @bomb_interaction_events and  users_self != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[11] = 1

    return context_vector
