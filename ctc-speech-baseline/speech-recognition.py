import time
import tensorflow as tf
import os
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc


SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 reserved for space

num_features = 13
num_classes = 28  # [a..z], <space>, <blank>

# training hyper-parameters
num_epochs = 200
num_hidden = 50
num_layers = 1  # lstm layers
batch_size = 1  # ????
leaning_rate = 0.01
momentum = 0.9

# dataset
num_examples = 1
mini_batch_size = 1

# load training audio
audio_filename = '1.wav'
target_filename = '1.txt'


def data_path(file_name):
    dir = os.path.dirname(__file__)
    return os.path.join(dir, '..', 'data', file_name)


def char2index(x):
    return SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape


fs, audio = wav.read(data_path(audio_filename))

# extract mfcc spectogram
spectogram = mfcc(audio, samplerate=fs)  # number of frames? default config?
print(f"input mfcc -> {spectogram.shape}")

# input array
train_inputs = np.asarray(spectogram[np.newaxis, :])
train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
train_steps = [spectogram[1]]

# load target sequence
with open(data_path(target_filename), 'r') as f:
    line = f.readlines()[-1]
    # clean target
    target_text = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
    target_text = target_text.replace(' ', '  ')
    print(f'clean target sequence -> {target_text}')
    targets = target_text.split(' ')
    print(f'targets (notice the importance of "silence gaps")-> {targets}')
    print(f'targets len -> {len(targets)}')

# Adding blank label
targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

# char 2 index
targets = np.asarray([char2index(x) for x in targets])

# Creating sparse representation to feed the placeholder
train_targets = sparse_tuple_from([targets])

val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_steps


print('done')

# prepare rnn input array











