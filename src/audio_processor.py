import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
sys.path.append(ROOT)
    
import numpy as np 
import cv2
import librosa
import warnings
import scipy
from scipy import signal

import time
from datetime import timedelta as td

# ----------------------------------------------------------------------
def rand_num(val): # random [-val, val]
    return (np.random.random()-0.5)*2*val

def integral(arr):
    sums = [0]*len(arr)
    for i in range(1, len(arr)):
        sums[i] = sums[i-1] + arr[i]
    return sums

def filter_by_average(arr, N):
    cumsum = np.cumsum(np.insert(arr, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N  

# ----------------------------------------------------------------------
# Time domain processings
def resample_audio(data, sample_rate, new_sample_rate):
    data = librosa.core.resample(data, sample_rate, new_sample_rate)
    return data, new_sample_rate

def filter_audio_by_average(data, sample_rate, window_seconds):
    window_size = int(window_seconds * sample_rate)
    
    sums = integral(data)
    res = [0]*len(data)
    for i in range(1, len(data)):
        prev = max(0, i - window_size)
        res[i] = (sums[i] - sums[prev]) / (i - prev)        
    return res


def remove_silent_prefix_by_freq_domain(
        data, sample_rate, n_mfcc, threshold, padding_s=0.2,
        return_mfcc=False):
    
    # Compute mfcc, and remove silent prefix
    mfcc_src = compute_mfcc(data, sample_rate, n_mfcc)
    mfcc_new = remove_silent_prefix_of_mfcc(mfcc_src, threshold, padding_s)

    # Project len(mfcc) to len(data)
    l0 = mfcc_src.shape[1]
    l1 = mfcc_new.shape[1]
    start_idx = int(data.size * (1 - l1 / l0))
    new_audio = data[start_idx:]
    
    # Return
    if return_mfcc:        
        return new_audio, mfcc_new
    else:
        return new_audio
        
def remove_silent_prefix_by_time_domain(
    data, sample_rate, threshold=0.25, window_s=0.1, padding_s=0.2):
    window_size = int(window_s * sample_rate)
    trend = filter_by_average(abs(data), window_size)
    start_idx = np.argmax(trend > threshold)
    start_idx = max(0, start_idx + window_size//2 - int(padding_s*sample_rate))
    return data[start_idx:]



# ----------------------------------------------------------------------
def compute_mfcc(data, sample_rate, n_mfcc=12):
    # Extract MFCC features
    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sample_rate,
        n_mfcc=n_mfcc
        # https://dsp.stackexchange.com/questions/28898/mfcc-significance-of-number-of-features
    )
    return mfcc 

def compute_log_specgram(audio, sample_rate, window_size=20,
                step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    MAX_FREQ = 9999999999999
    for i in range(len(freqs)):
        if freqs[i] > MAX_FREQ:
            break 
    freqs = freqs[0:i]
    spec = spec[:, 0:i]
    return freqs, np.log(spec.T.astype(np.float32) + eps)

def mfcc_to_image(mfcc, row=200, col=400,
                mfcc_min=-200, mfcc_max=200):
    ''' Convert mfcc to an image by converting it to [0, 255]'''
    
    # Rescale
    mfcc_img = 256 * (mfcc - mfcc_min) / (mfcc_max - mfcc_min)
    
    # Cut off
    mfcc_img[mfcc_img>255] = 255
    mfcc_img[mfcc_img<0] = 0
    mfcc_img = mfcc_img.astype(np.uint8)
    
    # Resize to desired size
    img = cv2.resize(mfcc_img, (col, row))
    return img

def pad_mfcc_to_fix_length(mfcc, goal_len=100, pad_value=-200):
    feature_dims, time_len = mfcc.shape
    if time_len >= goal_len:
        mfcc = mfcc[:, :-(time_len - goal_len)] # crop the end of audio
    else:
        n = goal_len - time_len
        zeros = lambda n: np.zeros((feature_dims, n)) + pad_value
        if 0: # Add paddings to both side
            n1, n2 = n//2, n - n//2
            mfcc = np.hstack(( zeros(n1), mfcc, zeros(n2)))
        else: # Add paddings to left only
            mfcc = np.hstack(( zeros(n), mfcc))
    return mfcc

def calc_histogram(mfcc, bins=10, binrange=(-50, 200), col_divides=5): 
    ''' Function:
            Divide mfcc into $col_divides columns.
            For each column, find the histogram of each feature (each row),
                i.e. how many times their appear in each bin.
        Return:
            features: shape=(feature_dims, bins*col_divides)
    '''
    feature_dims, time_len = mfcc.shape
    cc = time_len//col_divides # cols / num_hist = size of each hist
    def calc_hist(row, cl, cr):
        hist, bin_edges = np.histogram(mfcc[row, cl:cr], bins=bins, range=binrange)
        return hist/(cr-cl)
    features = []
    for j in range(col_divides):
        row_hists = [calc_hist(row, j*cc, (j+1)*cc) for row in range(feature_dims) ]
        row_hists = np.vstack(row_hists) # shape=(feature_dims, bins)
        features.append(row_hists)
    features = np.hstack(features)# shape=(feature_dims, bins*col_divides)
    return features 


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False):
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal

output = removeNoise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip,verbose=True,visual=True)



if __name__ == "__main__":
    pass
