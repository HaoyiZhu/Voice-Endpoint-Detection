import os

import math
import numpy as np
import json
import wave
import scipy

from scipy.io import wavfile
import scipy.signal as signal
import scipy.fftpack as fftpack

from tqdm import tqdm

# from python_speech_features import mfcc, logfbank

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    print('The data is successfully saved into', filename)

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def _read_wav_scipy(filename, dataset='dev', normalize=True, silence=True):
    if filename.split('.')[-1] != 'wav':
        filename = filename + '.wav'
        if not os.path.exists(filename):
            filename = os.path.join('./wavs/' + dataset, filename)
        if not silence:
            print('Trying to read .wav file from', filename)

    sample_rate, data = wavfile.read(filename)
    data = data.astype(np.float32)

    if normalize:
        data = data * 1.0 / (max(abs(data)))    # normalize

    return sample_rate, data

def _read_wav_wave(filename, dataset='dev', normalize=True, silence=True):
    if filename.split('.')[-1] != 'wav':
        filename = filename + '.wav'
        if not os.path.exists(filename):
            filename = os.path.join('./wavs/' + dataset, filename)
        if not silence:
            print('Trying to read .wav file from', filename)

    wav_file = wave.open(filename,'r')
    # num_channel = wav_file.getnchannels()
    # sample_width = wav_file.getsampwidth()
    # frame_rate = wav_file.getframe_rate()
    num_frames = wav_file.getnframes()

    data = wav_file.readframes(num_frames)
    data = np.fromstring(data, dtype=np.float32)

    if normalize:
        data = data * 1.0 / (max(abs(data)))    # normalize

    return num_frames, data

def read_wav(filename, dataset='dev', normalize=False, use_scipy=True, silence=True):
    if use_scipy:
        return _read_wav_scipy(filename, dataset, normalize, silence=silence)
    else:
        return _read_wav_wave(filename, dataset, normalize, silence=silence)

def cut_to_frame(data, frame_size=0.032, frame_shift=0.008, use_hamming=True):
    nw = int(frame_size * 16000)
    inc = int(frame_shift * 16000)
    data_length = len(data)

    if data_length <= nw:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * data_length - nw + inc) / inc))

    pad_length = int((nf - 1) * inc + nw)
    zeros = np.zeros((pad_length - data_length))
    pad_data = np.concatenate((data, zeros))

    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_data[indices]

    if use_hamming:
        winfunc = signal.hamming(nw)
        win = np.tile(winfunc, (nf, 1))
        return frames * win
    else:
        return frames

def ZeroCR(data, frame_size=0.032, frame_shift=0.008):
    data_len = len(data)
    overlap = int((frame_size - frame_shift) * 16000)
    frame_size = int(frame_size * 16000)
    step = int(frame_shift * 16000)
    frame_num = math.ceil((data_len - overlap) / step)

    zcr = np.zeros((frame_num,1))
    for i in range(frame_num):
        cur_frame = data[np.arange(i * step, min(i * step + frame_size, data_len))]
        cur_frame = cur_frame - np.mean(cur_frame) # zero-justified
        zcr[i] = sum(cur_frame[0:-1] * cur_frame[1::] <= 0)
    return zcr

def norm_1d(data):
    return (data - np.mean(data)) / np.max(data)

def _extract_features(data, frame_size=0.032, frame_shift=0.008):
    data_frame = cut_to_frame(data, frame_size=frame_size, frame_shift=frame_shift,
                              use_hamming=False)    # frame_num * frame_size

    data_norm = cut_to_frame(data * 1.0 / (max(abs(data))), frame_size=frame_size,
                             frame_shift=frame_shift, use_hamming=True)    # frame_num * frame_size

    zcr = ZeroCR(data, frame_size=frame_size, frame_shift=frame_shift)      # frame_num * 1

    fft_data = np.abs(fftpack.fft(data_norm)).astype(np.float32)    # frame_num * frame_size

    energy = np.sum((data_frame ** 2), axis=1).reshape(-1,1)    # frame_num * 1
    amplitude = np.sum(np.abs(data_frame),axis=1).reshape(-1,1)    # frame_num * 1

    # mfcc_features = mfcc(wav_data, 16000)
    # filterbank_features = logfbank(wav_data, 16000)
    # print(data_norm.shape, fft_data.shape, zcr.shape, energy.shape, amplitude.shape)
    all_features = np.concatenate((data_norm, fft_data, zcr, energy, amplitude), axis=1)

    return all_features

def sklearn_dataset_for_task_1(label, frame_size=0.032, frame_shift=0.008, silence=True,
                                        features_path='./task1/train_features.npy',
                                        target_path='./task1/train_target.npy'):
    if os.path.exists(features_path) and os.path.exists(target_path):
        print('lazy loading features and targets...')
        features = np.load(features_path)
        target = np.load(target_path)
        return features, target

    print('loading features and targets...')

    file_names = list(label.keys())
    target = list(label.values())
    features = []

    for i in tqdm(range(len(file_names))):
        _, wav_data = read_wav(file_names[i], 'dev', silence=silence)

        feature = _extract_features(wav_data, frame_size=frame_size, frame_shift=frame_shift)
        frame_num = feature.shape[0]
        label_pad = np.pad(target[i], (0, np.maximum(frame_num - len(target[i]), 0)))[:frame_num]
        target[i] = label_pad
        features.append(feature)

    target = np.concatenate(target)
    features = np.concatenate(features)

    if not os.path.exists(os.path.dirname(features_path)):
        os.mkdir(os.path.dirname(features_path))

    print('saving features and targets...')

    np.save(features_path[:-4], features)
    np.save(target_path[:-4], target)

    return features, target
    