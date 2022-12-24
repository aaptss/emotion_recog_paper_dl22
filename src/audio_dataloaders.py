from data_csgo_tournament.csgo_tournament_metadata import SAMPLING_RATE, PCS_LEN_SEC
from .demfile_parser import get_context_vector
from .audio_dataset_splitter import handle_audio_loading

import librosa
import soundfile as sf
from pydub import AudioSegment

import torch
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import time
from tqdm.notebook import tqdm


class BaseAudioSignalDataset(Dataset):
    def __init__(
                    self, 
                    data_list, 
                    use_game_context=False,
                    path_to_processed_csgo_data=None, 
                    sampling_rate=SAMPLING_RATE,
                    transform=None,
                    **ignored_kwargs
                ):

        assert (use_game_context and (path_to_processed_csgo_data is not None) or not use_game_context) 
        self.data_list = data_list
        self.sampling_rate = sampling_rate
        self.use_game_context = bool(use_game_context)
        self.path_to_processed_csgo_data = path_to_processed_csgo_data
        self.transform = transform

        if self.use_game_context:
            print("Preparing game context vectors")
            start = time.time()
            for _, _, info in tqdm(self.data_list):
                contexts = get_context_vector(*info, self.path_to_processed_csgo_data)
            print(f"Finished in {(time.time() - start) / 60:.2f} minutes")

    def __len__(self):
        return len(self.data_list)

    def load_wav(self, path):
        sfdata, sr = sf.read(path, always_2d=True, dtype='float32')  # load audio to 1d array
        sound_1d_array = sfdata[:,0]/2 + sfdata[:,1]/2 
        assert sr == self.sampling_rate
        return sound_1d_array

    def extract_features(self, path):
        sample = self.load_wav(path)
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample)

    def __getitem__(self, index):
        item = self.data_list[index]
        x = self.extract_features(item[0])
        y = item[1] - 1

        if self.use_game_context:
            ctx = torch.from_numpy(get_context_vector(*item[2], self.path_to_processed_csgo_data)).float()
        else:
            ctx = torch.tensor([])
        return x, ctx, y


class BaseSpectrogramDataset(BaseAudioSignalDataset):
    def __init__(self,
                 data_list,
                 path_to_processed_csgo_data,
                 use_game_context=True,
                 sampling_rate=SAMPLING_RATE,
                 window_size=512,
                 transform=None,
                 **ignored_kwargs):
        self.window_size = window_size
        self.hop_len = self.window_size // 2
        self.path_to_processed_csgo_data = path_to_processed_csgo_data

        self.window_weights = np.hanning(self.window_size)[:, None]
        self.transform=transform
        super().__init__(data_list, use_game_context, self.path_to_processed_csgo_data, sampling_rate)

    @staticmethod
    def __visualize__(spec):
        ax = sns.heatmap(spec)
        ax.invert_yaxis()

    def extract_features(self, path):
        _track = self.load_wav(path)

        spec = self.calculate_all_windows(_track)
        spec_offset = spec - np.min(spec)
        spec_offset = 255 * spec_offset/np.max(spec_offset)
        spec_offset = np.round(spec_offset)

        channel_amt = 13 if self.use_game_context else 1
        out = torch.zeros(channel_amt, *spec.shape)

        # out[0] = torch.tensor(spec_offset, dtype=torch.uint8)
        out[0] = torch.tensor(spec_offset)
        return out

    def __getitem__(self, index):
        item = self.data_list[index]
        x = self.extract_features(item[0])
        if self.transform:
            x = self.transform(x)
        y = item[1] - 1

        if self.use_game_context:
            ctx = get_context_vector(*item[2], self.path_to_processed_csgo_data)
            for event, val in enumerate(ctx):
                x[event+1] = 255 * torch.ones_like(x[0]) if val else torch.zeros_like(x[0])
        else:
            ctx = torch.tensor([])
        return x, ctx, y

    def calculate_all_windows(self, audio):
        """
        For a typical speech recognition task, 
        a window of 20 to 30ms long is recommended.
        The overlap can vary from 25% to 75%.
        it is kept 50% for speech recognition.
        """
        truncate_size = (len(audio) - self.window_size) % self.hop_len
        audio = audio[:len(audio) - truncate_size]

        nshape = (self.window_size, (len(audio) - self.window_size) // self.hop_len)
        nhops = (audio.strides[0], audio.strides[0] * self.hop_len)

        windows = np.lib.stride_tricks.as_strided(audio,
                                                  shape=nshape,
                                                  strides=nhops)

        assert np.all(windows[:, 1] == audio[self.hop_len:(self.hop_len + self.window_size)])

        yf = np.fft.rfft(windows * self.window_weights, axis=0)
        yf = np.abs(yf) ** 2

        scaling_factor = np.sum(self.window_weights ** 2) * self.sampling_rate
        yf[1:-1, :] *= (2. / scaling_factor)
        yf[(0, -1), :] /= scaling_factor

        xf = float(self.sampling_rate) / self.window_size * np.arange(yf.shape[0])

        indices = np.where(xf <= self.sampling_rate // 2)[0][-1]
        return np.log(yf[:indices, :] + 1e-16)


# for ML methods
def get_data(data_list, use_game_context, path_to_processed_csgo_data):
    amt_of_samples = SAMPLING_RATE * PCS_LEN_SEC

    X = []
    Y = []
    ctx_ = []

    for item in data_list:
        sound_1d_array, _ = librosa.load(item[0])
        if sound_1d_array.size < amt_of_samples:
            offset = amt_of_samples - sound_1d_array.size
            sound_1d_array = np.pad(sound_1d_array, (0, offset))

        X.append(sound_1d_array)
        y_onehot = create_onehot_tensor(item[1]).numpy()
        Y.append(y_onehot)
        if use_game_context:
            ctx_.append(get_context_vector(*item[2], path_to_processed_csgo_data))

    if use_game_context:
        return np.array(X), np.array(ctx_), np.array(Y)

    return np.array(X), np.array(Y)


def get_dataloader(file_path, 
                   path_to_audio, 
                   path_to_splitted_audio, 
                   path_to_processed_csgo_data,
                   DatasetClass,
                   splitsize=0.8,
                   transform=None,
                   use_game_context=False,
                   batch_size=32,
                   num_workers=8,
                   train_list=None,
                   val_list=None
                   ):
    if train_list is None:
        val_list = None
        train_list, val_list = handle_audio_loading(file_path, path_to_audio, path_to_splitted_audio, splitsize)

    print('\nPrepare train dataset')
    train_dataset = DatasetClass(train_list, 
                                 path_to_processed_csgo_data=path_to_processed_csgo_data,
                                 use_game_context=use_game_context, 
                                 num_workers=num_workers,
                                 transform=transform)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers, 
                                  shuffle=True)

    print('Prepare val dataset')
    val_dataset = DatasetClass(val_list, 
                               path_to_processed_csgo_data=path_to_processed_csgo_data,
                               use_game_context=use_game_context, 
                               num_workers=num_workers,
                               transform=None)

    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)

    return train_dataloader, val_dataloader