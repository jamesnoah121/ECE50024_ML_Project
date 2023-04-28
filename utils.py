# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:31:20 2023

@author: Noah

# ML Project Utilities

"""

import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
from keras import models
from keras import layers

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np


import IPython.display as ipd


def play_audio_signal(x,sr):
    
    sd.play(x, sr)

def display_audio_signal(x,sr):
    librosa.display.waveplot(x, sr=sr)
    
def convert_audio_waveform_to_mel_spectrogram(x):
    
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    return X, Xdb


