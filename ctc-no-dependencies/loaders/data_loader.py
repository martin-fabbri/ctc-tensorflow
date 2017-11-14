"""
Defines a class that is used to load and featurize audio data
for training and testing.
"""

# todo: [ ] load mfccs and labels
# todo: [ ] load batches(mini batches)
# todo: [ ] split data into train, test, validation

import logging
import random
import os
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from utils import RANDOM_SEED, char_to_int_encode
from .audio_loader import mfcc_segment, normalize

logger = logging.getLogger(__name__)


# load training metadata
def _load_metadata_from_desc_file(desc_file, max_duration):
    """
    Reads metadata from dataset descriptor file.

    :param desc_file: Path to a JSON-line file that contains labels and
            paths to the audio files
    :param max_duration:In seconds, the maximum duration of
            utterances to train or test on
    :return:
    """
    logger.info(f"Loading dataset metadata from {desc_file}")
    audio_paths, texts = [], []
    with open(desc_file, encoding='utf-8-sig') as ymal_file:
        data = yaml.load(ymal_file)
        for spec in data:
            if float(spec['duration']) > max_duration:
                continue
            audio_paths.append(spec['key'])
            texts.append(spec['text'])
    return audio_paths, texts

class DataLoader(object):
    def __init__(self, config):
        """
        :param desc_file: Path to dataset description file
        :param max_freq:
        :param test_size:
        :param random_state:
        """
        self.dataset, self.audio, self.train = config.dataset, config.audio, config.train
        self.rng = random.Random(RANDOM_SEED)
        audio_paths, texts = _load_metadata_from_desc_file(self.dataset.path,
                                                           self.dataset.max_duration_seg)
        self.audio_train, self.audio_test, self.text_train, self.text_test = \
            train_test_split(audio_paths, texts,
                             test_size=self.dataset.test_size,
                             random_state=RANDOM_SEED)
        logger.info("DataLoader initialization complete.")

    def _prepare_placeholder(self, wav_file_placeholder, audio_clip, append_path):
        return {wav_file_placeholder: audio_clip if append_path is None else os.path.join(append_path, audio_clip)}

    # graph regions
    def _prepare_minibatch(self, audio_paths, texts, sess, mfcc, wav_file_placeholder, append_path):
        """
        Featurize a minibatch of audio
        :param audio_paths: List of paths to audio files
        :param texts: List of texts corresponding to the audio files
        :param sess: Active TensorFlow session used for feature extraction
        :param mfcc: MFCC extraction segment
        :param wav_file_placeholder: Data feed place holder
        :return:
        """
        # todo: need to implement zero pad?
        assert len(audio_paths) == len(texts), "Inputs and outputs to the network must be of the same number"
        # features = [normalize(sess.run(mfcc, self._prepare_placeholder(wav_file_placeholder, audio_clip, append_path))
        #                       for audio_clip in audio_paths)]
        features = [normalize(sess.run(mfcc, {wav_file_placeholder: audio_clip})) for audio_clip in audio_paths]
        labels = [np.asarray(char_to_int_encode(target_text)) for target_text in texts]
        return features, labels, texts

    def _iterate(self, audio_paths, texts, minibatch_size, sess, append_path):
        k_iters = int(np.ceil(len(audio_paths) / minibatch_size))
        logger.debug("Preparing {k_iters} minibatches iters")
        mfcc, wav_file_placeholder = mfcc_segment(self.audio)
        pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
        future = pool.submit(self._prepare_minibatch,
                             audio_paths[:minibatch_size],
                             texts[:minibatch_size],
                             sess,
                             mfcc,
                             wav_file_placeholder,
                             append_path)
        start = minibatch_size
        for i in range(k_iters - 1):
            wait([future])
            minibatch = future.result()
            # while the current minibatch is being consumed, prepare the next
            future = pool.submit(self._prepare_minibatch,
                                 audio_paths[start: start + minibatch_size],
                                 texts[start: start + minibatch_size],
                                 sess,
                                 mfcc,
                                 wav_file_placeholder,
                                 append_path)
            yield minibatch
            start += minibatch_size
        # wait on the last minibatch
        wait([future])
        minibatch = future.result()
        yield minibatch

    def iterate_training_batch(self, sess, append_path=None):
        logger.debug(f"Preparing training batch: minibatch_size {self.train.minibatch_size}")
        audio_paths, texts = self.audio_train, self.text_train
        if self.train.shuffle:
            temp = zip(audio_paths, texts)
            self.rng.shuffle(temp)
            audio_paths, texts = zip(*temp)
        return self._iterate(audio_paths, texts, self.train.minibatch_size, sess, append_path=None)

    def iterate_testing_batch(self, sess, minibatch_size=16, sort_by_duration=False, shuffle=False):
        pass

    def iterate_validation_batch(self, sess, minibatch_size=16, sort_by_duration=False, shuffle=False):
        pass
