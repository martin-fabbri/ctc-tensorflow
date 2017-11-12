"""
Defines a class that is used to load and featurize audio data
for training and testing.
"""

# todo: [ ] load mfccs and labels
# todo: [ ] load batches(mini batches)
# todo: [ ] split data into train, test, validation

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from sklearn.model_selection import train_test_split

from .audio_loader import mfcc_segment, normalize
from utils import RANDOM_SEED

logger = logging.getLogger(__name__)

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


def sort_by_duration(durations, audio_paths, texts):
    return


class DataLoader(object):
    def __init__(self, desc_file, test_size=None, max_duration=10.0):
        """
        :param desc_file: Path to dataset description file
        :param max_freq:
        :param test_size:
        :param random_state:
        """
        self.rng = random.Random(RANDOM_SEED)
        audio_paths, texts = load_metadata_from_desc_file(desc_file, max_duration)
        self.audio_train, self.audio_test, self.text_train, self.text_test = train_test_split(audio_paths, texts,
                                                                                              test_size=test_size,
                                                                                              random_state=self.rng)
        logger.info("DataLoader initialization complete.")

    def __call__(self, normalized_mfcc, wav_file_placeholder):
        self.build_batch_fetching_segment(normalized_mfcc, wav_file_placeholder)
        return self

    # graph regions
    def prepare_minibatch(self, audio_paths, texts, sess, mfcc, wav_file_placeholder):
        """ Featurize a minibatch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts), "Inputs and outputs to the network must be of the same number"
        features = [normalize(sess.run(mfcc, {wav_file_placeholder: a})) for a in audio_paths]
        labels = []
        return features, labels, texts

    def iterate(self, audio_paths, texts, minibatch_size, sess):
        k_iters = int(np.ceil(len(audio_paths) / minibatch_size))
        logger.debug("Preparing {k_iters} minibatches iters")
        mfcc, wav_file_placeholder = mfcc_segment(window_size_ms, window_stride_ms, dct_coefficient_count)
        pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
        future = pool.submit(self.prepare_minibatch,
                             audio_paths[:minibatch_size],
                             texts[:minibatch_size],
                             sess)
        start = minibatch_size
        for i in range(k_iters - 1):
            wait([future])
            minibatch = future.result()
            # While the current minibatch is being consumed, prepare the next
            future = pool.submit(self.prepare_minibatch,
                                 audio_paths[start: start + minibatch_size],
                                 texts[start: start + minibatch_size],
                                 sess
                                 )
            yield minibatch
            start += minibatch_size
        # Wait on the last minibatch
        wait([future])
        minibatch = future.result()
        yield minibatch

    def next_training_batch(self, sess, minibatch_size=16, sort_by_duration=False, shuffle=True):
        logger.debug(f"Preparing training batch: minibatch_size {minibatch_size}")
        durations, audio_paths, texts = self.train_durations, self.train_audio_paths, self.train_texts
        if shuffle:
            temp = zip(durations, audio_paths, texts)
            self.rng.shuffle(temp)
            durations, audio_paths, texts = zip(*temp)
        elif sort_by_duration:
            durations, audio_paths, texts = sort_by_duration(durations, audio_paths, texts)

        return self.iterate(audio_paths, texts, minibatch_size, sess)

    def next_testing_batch(self, sess, minibatch_size=16, sort_by_duration=False, shuffle=False):
        pass

    def next_validation_batch(self, sess, minibatch_size=16, sort_by_duration=False, shuffle=False):
        pass


