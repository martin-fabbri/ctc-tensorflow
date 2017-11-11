"""
Defines a class that is used to load and featurize audio data
for training and testing.
"""

# todo: load mfcc and labels
# todo: load batches(mini batches)
# todo: split data into train, test, validation

import numpy as np
import logging
import json
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


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
