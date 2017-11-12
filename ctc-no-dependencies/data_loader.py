"""
Defines a class that is used to load and featurize audio data
for training and testing.
"""

# todo: [ ] load mfccs and labels
# todo: [ ] load batches(mini batches)
# todo: [ ] split data into train, test, validation

import tensorflow as tf
import numpy as np
import logging
import json
from tensorflow.python.ops import io_ops
from sklearn.model_selection import train_test_split
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

logger = logging.getLogger(__name__)


# graph regions
def _load_wav_file_segment(filename):
    """
    Builds a TensorFlow graph segment that loads a .wav file
    This function should be called within an **Active** TensorFlow session
    :arg filename: Path to .wav file
    :arg sess:
    :return: Audio encoder node, Node's placeholders dict
    """
    wav_file_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_file_placeholder)
    # todo: do we need to pass sample rate?
    feed_dict = {wav_file_placeholder: filename}
    return contrib_audio.decode_wav(wav_loader, desired_channels=1), feed_dict


def _load_mfcc_segment(filename, window_size_ms, window_stride_ms, dct_coefficient_count):
    """
    Builds a TensorFlow graph segment that extract the MFCC fingerprints

    :param filename:
    :param sess:
    :param window_size_ms: time slice duration to estimate frequencies from
    :param dct_coefficient_count: How many output channels to produce per time slice
    :param sample_rate??:
    :return:
    """
    # todo: desired_samples = 16000  # todo: default should be 16K instead?
    wav_decoder, feed_dict = _load_wav_file_segment(filename)
    spectrogram = contrib_audio.audio_spectrogram(
        wav_decoder.audio,
        window_size=window_size_ms,  # for efficiency make it a power of 2
        stride=window_stride_ms,
        magnitude_squared=False
    )
    mfcc = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate, dct_coefficient_count=dct_coefficient_count)
    return mfcc, feed_dict


# standalone audio processing
def load_wav_file(filename):
    with tf.Session(graph=tf.Graph()) as sess:
        wav_decoder, feed_dict = _load_wav_file_segment(filename)
        return sess.run(wav_decoder, feed_dict).audio.flatten()


def load_mfcc(filename, window_size_ms=550, window_stride_ms=350, dct_coefficient_count=13):
    """
    Builds a TensorFlow graph segment that extract the MFCC fingerprints

    :param filename:
    :param sess:
    :param window_size_ms: time slice duration to estimate frequencies from
    :param window_stride_ms:
    :param dct_coefficient_count: How many output channels to produce per time slice
    :param sample_rate??:
    :return:
    """
    with tf.Session(graph=tf.Graph()) as sess:
        mfcc, feed_dict = _load_mfcc_segment(filename, window_size_ms, window_stride_ms, dct_coefficient_count)
        return sess.run(mfcc, feed_dict)


# load training metadata
def load_metadata_from_desc_file(desc_file, max_duration):
    """
    Reads metadata from dataset descriptor file.

    :param desc_file: Path to a JSON-line file that contains labels and
            paths to the audio files
    :param max_duration:In seconds, the maximum duration of
            utterances to train or test on
    :return:
    """
    logger.info(f"Loading dataset metadata from {desc_file}")
    audio_paths, durations, texts = [], [], []
    try:
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                except Exception as e:
                    logger.warning(f"Error reading line #{line_num}: {json_line}")
                    logger.warning(str(e))
    except Exception:
        raise Exception("Descriptor file not found.")

    return audio_paths, texts


class DataLoader(object):
    def __init__(self, desc_file, window_ms=20, max_freq=8000, max_duration=10.0, test_size=None, random_state=None):
        """
        :param desc_file: Path to dataset description file
        :param window_ms: MFCC window size in milliseconds
        :param max_freq:
        :param test_size:
        :param random_state:
        """
        audio_paths, texts = load_metadata_from_desc_file(desc_file, max_duration)
        self.audio_train, self.audio_test, self.text_train, self.text_test = train_test_split(audio_paths, texts,
                                                                                              test_size=test_size,
                                                                                              random_state=random_state)
        logger.info("DataLoader initialization complete.")

    def featurize(self, audio_wav):
        """
        for a given audio file
        """
        pass

    def prepare_minibatch(self, audio_paths, texts):
        """ Featurize a minibatch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts), "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        # Calculate the features for each audio clip, as the log of the
        # Fourier Transform of the audio
        features = [self.featurize(a) for a in audio_paths]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((mb_size, max_length, feature_dim))
        y = []
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            feat = self.normalize(feat)  # Center using means and std
            x[i, :feat.shape[0], :] = feat
            label = text_to_int_sequence(texts[i])
            y.append(label)
            label_lengths.append(len(label))
        y = None
        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'y': y,  # list(int) Flattened labels (integer sequences)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths  # list(int) Length of each label
        }

    def iterate(self, audio_paths, texts, minibatch_size):
        k_iters = int(np.ceil(len(audio_paths) / minibatch_size))
        logger.info(f"Iters: {k_iters}")
        start = minibatch_size
        for i in range(k_iters - 1):
            # While the current minibatch is being consumed, prepare the next
            x = self.prepare_minibatch(
                audio_paths[start: start + minibatch_size],
                texts[start: start + minibatch_size]
            )
            yield x
            start += minibatch_size

    def load_train_data(self, desc_file):
        load_metadata_from_desc_file(desc_file, 'train')

    def next_training_batch(self, minibatch_size=16, sort_by_duration=False, shuffle=True):
        """
        :param minibatch_size:
        :param sort_by_duration:
        :param shuffle:
        :return:
        """
        # return self.iterate(audio_paths, texts, minibatch_size)
        yield 1
        yield 2
        yield 3

    def next_testing_batch(self, minibatch_size=16, sort_by_duration=False, shuffle=True):
        yield 1
        yield 2
        yield 3

    def next_validation_batch(self, minibatch_size=16, sort_by_duration=False, shuffle=True):
        yield 1
        yield 2
        yield 3
