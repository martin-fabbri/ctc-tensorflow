import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import io_ops
import matplotlib.pyplot as plt
from input_audio import load_wav_file
from input_audio import load_mfcc

file_name = "1.wav"
data_dir = "data"
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, "..", data_dir, file_name)
mfcc = load_mfcc(file_path)
mfccs = np.transpose(mfcc[0])

# plt.plot(mfcc[0])


from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

rate, sig = wav.read(file_path)
#mfccs = mfcc(sig, rate)


import librosa
import librosa.display
y, sr = librosa.load(file_path)
#mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

# Visualize the MFCC series
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
