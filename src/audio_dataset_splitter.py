from data_csgo_tournament.csgo_tournament_metadata import EMOTION_WITH_KEYS, NUMBER_OF_PLAYERS, MATCH_LEN
from .audio_splitter import split_audio
import glob, os, sys, pathlib
from functools import lru_cache

import json
import pandas as pd
import numpy as np

from collections import Counter
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import time
from datetime import datetime



# simple majority vote to determine true label
def find_majority(est_with_str):
    votes = [i[0] for i in est_with_str]
    strength = [i[1] for i in est_with_str]
    vote_count = Counter(votes)
    most_ = vote_count.most_common(1)
    if (most_[0][1] >= 2) and (most_[0][0] > 0):
        ids = [ind for ind in range(len(votes)) if votes[ind] == most_[0][0]]
        mean_str = round(np.array([strength[i] for i in ids]).mean(), 1)
        return [most_[0][0], mean_str]
    else:
        if (most_[0][0] == 0) and (most_[0][1] == 2):
            return [max(votes), max(strength)]
        return [-1, -1]


def get_emotions(df):
    player_emt = df.copy()
    for i in range(1, 4):
        player_emt[f'est_{i}'] = list(zip(player_emt[f'emt_est_{i}'], player_emt[f'str_emt_est_{i}']))

    ss = []
    for i in player_emt[['est_1', 'est_2', 'est_3']].values:
        ss.append(find_majority(i))
    player_emt['emt'] = np.asarray(ss)[:, 0]
    player_emt['emt'] = player_emt['emt'].apply(int)
    player_emt['str_emt'] = np.asarray(ss)[:, 1]

    emt_in_time = {}
    for i in player_emt[['start', 'emt']].query('emt>0').emt.unique():
        emt_in_time[i] = [j[0] for j in player_emt[['start', 'emt']].query('emt>0').values if j[1] == i]
    return emt_in_time


@lru_cache(None)
def get_dict_with_emotions(file_path):  # keys: match -> n_round -> num_player value: dataframe
    all_emt_dict = {}
    col_emt_names = ["start", "end", "emt_est_1", "str_emt_est_1", "emt_est_2", "str_emt_est_2", "emt_est_3",
                     "str_emt_est_3"]
    for n_match, rounds_in_match in MATCH_LEN.items():
        match_emt_cl = {}
        for n_round in range(1, rounds_in_match + 1):
            players_emt = {}
            res_emt = {}
            for num_player in range(NUMBER_OF_PLAYERS):
                if glob.glob(file_path + f'/{num_player}_match{n_match}_round{n_round}.csv'):
                    str_path = glob.glob(file_path + f'/{num_player}_match{n_match}_round{n_round}.csv')[0]
                    # try:
                        # df_s = pd.read_csv(str_path, names=col_emt_names)
                    # except UnicodeDecodeError:
                    #     if platform.system() == 'Windows':
                    #         os.system("notepad " + str_path)
                    #     else:
                    #         os.system("nano " + str_path)
                    df_s = pd.read_csv(str_path, names=col_emt_names)
                    res_emt[num_player] = df_s
            match_emt_cl[n_round] = res_emt
        all_emt_dict[n_match] = match_emt_cl
    return all_emt_dict


def handle_emotion_data(file_path):
    def display_emt(full_dict_with_emt):
        print('Emotion statistic:')
        total = 0
        for key, value in full_dict_with_emt.items():
            emot_amt = np.array([len(i[3]) for i in value]).sum()
            print(key, '-', emot_amt, '   ', EMOTION_WITH_KEYS[key])
            total += emot_amt
        print(f'{total} amount of samples in total')


    all_emt_dict = get_dict_with_emotions(file_path)
    full_emt = {}  # key=emotion_type: value=[n_player,n_match,n_round,start_time]
    
    for key in EMOTION_WITH_KEYS.keys():
        full_emt[key] = []

    for n_match, rounds_in_match in MATCH_LEN.items():
        for n_round in range(1, rounds_in_match + 1):
            for n_player in range(NUMBER_OF_PLAYERS):
                if all_emt_dict.get(n_match) is not None and \
                        all_emt_dict.get(n_match).get(n_round) is not None and \
                        all_emt_dict.get(n_match).get(n_round).get(n_player) is not None:
                    emt_in_time = get_emotions(all_emt_dict.get(n_match).get(n_round).get(n_player))
                    for e, start_time in emt_in_time.items():
                        full_emt[e].append([n_player, n_match, n_round, start_time])
    display_emt(full_emt)
    return full_emt


def split_annotations(full_emt, path_to_audio, splitsize):
    train_annotations_list = []
    val_annotations_list = []

    X = []
    y = []
    info = []
    val_size = 1 - splitsize

    for key_emt, values in tqdm(full_emt.items()):
        for vals in tqdm(values):
            n_player, n_match, n_round, list_start_time = vals
            for start_time in list_start_time:
                file = glob.glob(
                    path_to_audio + f'/{n_player}_match{n_match}_round{n_round}_{start_time}*.wav')
                X.append(file[0])
                y.append(key_emt)
                info.append([n_player, n_match, n_round, start_time])

    X_train, X_val, y_train, y_val, info_train, info_val = train_test_split(X, y, info, 
        test_size=val_size,
        random_state=42, shuffle=True, stratify=y)

    train_annotations_list = list(zip(X_train, y_train, info_train))
    val_annotations_list = list(zip(X_val, y_val, info_val))

    return train_annotations_list, val_annotations_list 


class NpEncoder(json.JSONEncoder):
    """
    support class to reinforce JSON.dump handling as it's unfamiliar with np data types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def handle_audio_loading(file_path, path_to_audio, path_to_splitted_audio, splitsize=0.8):
    if not os.path.exists(path_to_splitted_audio):
        print('Start splitting audio to ', path_to_splitted_audio)
        os.makedirs(path_to_splitted_audio)
        split_audio(path_to_audio, path_to_splitted_audio)
    else:
        print(f"{path_to_splitted_audio} exists; skip splitting")
    try:
        with open("prepared.json", "r") as file:
            print('Grabbing audio loading detais from the .json file')
            train_list, val_list = json.load(file)
    except Exception as e:
        print(f'Something went wrong while trying to grab data from .json: {e}')
        print('Commencing audio annotations splitting')
        full_dict_with_emt = handle_emotion_data(file_path)

        print('\nPrepare train and val lists')
        start = time.time()
        train_list, val_list = split_annotations(full_dict_with_emt, path_to_splitted_audio, splitsize)

        print("It took: ", round((time.time() - start) / 60, 2), " minutes")
        print('train size: ',len(train_list))
        print('val size: ', len(val_list))

        if os.path.exists("prepared.json"):
            print("json already exists")
        else:
            with open("prepared.json", "a+") as file:
                json.dump([train_list, val_list], file, cls=NpEncoder)
    return train_list, val_list