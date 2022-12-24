import os, pathlib, fnmatch
from pydub import AudioSegment

from data_csgo_tournament.csgo_tournament_metadata import DATASET_ROOT


# splitting audio recordings in 3sec interval according to LCB_.csv
def split_audio(path_to_audio=f'./{DATASET_ROOT}/LCB_data_snd_devided', path_to_splitted_audio=f'./{DATASET_ROOT}/LCB_audio'):
    for path in pathlib.Path(path_to_audio).iterdir():
        if path.is_dir():
            player_number = str(path)[str(path).rfind('/') + 1:].split('_')[0]
            for path_in in pathlib.Path(path).iterdir():
                str_path = str(path_in)
                name = str_path[str_path.rfind('/') + 1:]
                sound = AudioSegment.from_wav(path_in)
                i = 0
                while i <= len(sound):
                    if i + 3000 > len(sound):
                        cut = sound[i:len(sound) + 1]
                        if not os.path.exists(path_to_splitted_audio +
                                              '/' +
                                              player_number + '_' +
                                              name[:-4] + f'_{i}_{len(sound)}.wav'):
                            cut.export(path_to_splitted_audio +
                                       '/' +
                                       player_number + '_' +
                                       name[:-4] + f'_{i}_{len(sound)}.wav',
                                       format="wav")
                        break
                    cut = sound[i:i + 3000]
                    if not os.path.exists(path_to_splitted_audio +
                                          '/' +
                                          player_number + '_' + name[:-4] + f'_{i}_{i + 3000}.wav'):
                        cut.export(path_to_splitted_audio +
                                   '/' +
                                   player_number + '_' +
                                   name[:-4] + f'_{i}_{i + 3000}.wav',
                                   format="wav")
                    i += 3000


def handle_audio_splitting(path_to_audio=f'./{DATASET_ROOT}/LCB_audio', path_to_splitted_audio=f'./{DATASET_ROOT}/LCB_data_snd_devided'):
    file_cnt = 0
    if os.path.exists(path_to_splitted_audio):
        file_cnt = len(fnmatch.filter(os.listdir(path_to_splitted_audio), '*.wav'))
        print(f"{path_to_splitted_audio} exists. It contains {file_cnt} .wav files.")
    if file_cnt > 5:
        return
    print('Start splitting audio to ', path_to_splitted_audio)
    os.makedirs(path_to_splitted_audio)
    split_audio(path_to_audio, path_to_splitted_audio)
    print('Finish splitting\n')