import numpy as np
from scipy.io.wavfile import read
import torch
import librosa

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path):
    # scipy is buggy
    # sampling_rate, data = read(full_path)
    data, sampling_rate = librosa.load(full_path)
    print("Loading %s..." % full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    return torch.autograd.Variable(x)
