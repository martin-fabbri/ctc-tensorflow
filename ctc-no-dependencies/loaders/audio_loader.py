"""
Defines a class that is used to load and featurize audio data
for training and testing.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio


def normalize(feature, eps=1e-14):
    return (feature - np.mean(feature)) / (np.std(feature) + eps)

#window_ms=20, max_freq=8000, max_duration=10.0
# :param window_ms: MFCC window size in milliseconds
# :param max_freq:
# :param max_duration:


# graph regions
# todo: do we need to pass sample rate?
def load_wav_file_segment():
    """
    Builds a TensorFlow graph segment that loads a .wav file
    This function should be called within an **Active** TensorFlow session
    :arg filename: Path to .wav file
    :arg sess:
    :return: Audio encoder node
    """
    wav_file_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_file_placeholder)
    return contrib_audio.decode_wav(wav_loader, desired_channels=1), wav_file_placeholder


# todo: desired_samples = 16000  # todo: default should be 16K instead?
def mfcc_segment(window_size_ms, window_stride_ms, dct_coefficient_count):
    """
    Builds a TensorFlow graph segment that extract the MFCC fingerprints

    :param filename:
    :param window_size_ms: time slice duration to estimate frequencies from
    :param dct_coefficient_count: How many output channels to produce per time slice
    :param sample_rate??:
    :return:
    """
    wav_decoder, wav_file_placeholder = load_wav_file_segment()
    spectrogram = contrib_audio.audio_spectrogram(
        wav_decoder.audio,
        window_size=window_size_ms,  # for efficiency make it a power of 2
        stride=window_stride_ms,
        magnitude_squared=False
    )
    return contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate,
                              dct_coefficient_count=dct_coefficient_count), wav_file_placeholder


# standalone audio processing
def load_wav_file(filename):
    with tf.Session(graph=tf.Graph()) as sess:
        wav_decoder, wav_file_placeholder = load_wav_file_segment()
        feed_dict = {wav_file_placeholder: filename}
        return sess.run(wav_decoder, feed_dict).audio.flatten()


def load_mfcc(filename, window_size_ms=550, window_stride_ms=350, dct_coefficient_count=13):
    with tf.Session(graph=tf.Graph()) as sess:
        mfcc, wav_file_placeholder = mfcc_segment(window_size_ms, window_stride_ms, dct_coefficient_count)
        feed_dict = {wav_file_placeholder: filename}
        return sess.run(mfcc, feed_dict)
