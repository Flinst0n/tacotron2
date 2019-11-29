#!python3
# -*- coding: utf-8 -*-
# pylint: disable=W0312, line-too-long, C0103
# pylama: noqa
# flake8: noqa

import sys
sys.path.append('waveglow-mirror/')

import time

import librosa
import numpy as np
import torch
from torch.nn import functional as F

from audio_processing import griffin_lim
from hparams import create_hparams
from layers import STFT, TacotronSTFT
from model import Tacotron2
from text import text_to_sequence
from train import load_model


torch.set_num_threads(4)

hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
_ = model.cpu().eval()

waveglow_path = 'waveglow_256channels.pt'
waveglow = torch.load(waveglow_path, map_location='cpu')['model']
waveglow.cpu().eval()
# for k in waveglow.convinv:
#    k.float()

for m in waveglow.modules():
    if 'Conv' in str(type(m)):
        setattr(m, 'padding_mode', 'zeros')
        # print(m)


def generate_texts(texts):
    audios = None
    mel_outputs_postnet = []
    max_len = 0
    # tacotron
    start = time.time()
    for text in texts:

        print("Calculating %s" % (text[:10]))
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()

        _, mel, _, _ = model.inference(sequence)
        mel_outputs_postnet.append(mel)

        if mel.shape[2] > max_len:
            max_len = mel.shape[2]

    print("Padding data...")
    for i, mel in enumerate(mel_outputs_postnet):
        if mel.shape[2] < max_len:
            mel_outputs_postnet[i] = F.pad(mel, (0, max_len-mel.shape[2], 0, 0, 0, 0), value=float(min(mel.reshape(-1)).data.cpu()))

    tacotron_time = time.time() - start
    start = time.time()

    print("Concatenating...")
    mel_outputs_postnet = torch.cat(mel_outputs_postnet, 0)
    print(mel_outputs_postnet.shape)
    print("Synthesizing...")
    with torch.no_grad():
        all_audios = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    waveglow_time = time.time() - start

    print("Merging audio...")
    for audio in all_audios:
        if audios is not None:
            audios = np.append(audios, audio.data.cpu().numpy())
        else:
            audios = audio.data.cpu().numpy()

    audio_len = len(audios) / 22050

    print("Total calculation time %.2lf, Length %.2lf, Tacotron: %.2lfx-realtime, Waveglow: %.2lfx-realtime" % (tacotron_time + waveglow_time, audio_len, audio_len / tacotron_time, audio_len / waveglow_time))

    return audios


texts = ["A very warm welcome to my dear colleagues from Original Equipment Innovation.",
         "I am a new text to speech engine that is perceived to be really natural.",
         "I am based upon the so-called Tacotron Two model and a Waveglow synthesizer.",
        #  "I, hereby, apply as your new voice for nice chat!",
        #  "Currently I'm running offline on Alex's computer on an Intel Core i5 CPU, but I'm quite confident that I can also run on your smaller aircraft N U Cs.",
        #  "Alex Yasha and Eyekey will be happy to tell you more about me and my current capabilities!.",
         "Hmmm, well, on a second thought, kill all humans!"]
audios = generate_texts(texts)

librosa.output.write_wav("output.wav", audios, hparams.sampling_rate)
