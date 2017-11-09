"""Tests for data input for speech commands."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.platform import test
from audio_processor import prepare_processing_graph


class AudioProcessorTest(test.TestCase):
    def _getWavData(self):
        with self.test_session() as sess:
            sample_data = tf.zeros([1000, 2])
            wav_encoder = contrib_audio.encode_wav(sample_data, 16000)
            wav_data = sess.run(wav_encoder)
        return wav_data

    def _saveTestWavFile(self, filename, wav_data):
        with open(filename, "wb") as f:
            f.write(wav_data)

    def _saveWavFolders(self, root_dir, labels, how_many):
        wav_data = self._getWavData()
        for label in labels:
            dir_name = os.path.join(root_dir, label)
            os.mkdir(dir_name)
            for i in range(how_many):
                file_path = os.path.join(dir_name, "some_audio_%d.wav" % i)
                self._saveTestWavFile(file_path, wav_data)

    def _model_settings(self):
        return {
            "desired_samples": 160,
            "fingerprint_size": 40,
            "label_count": 4,
            "window_size_samples": 100,
            "window_stride_samples": 100,
            "dct_coefficient_count": 40,
        }

    def testPrepareProcessingGraph(self):
        tmp_dir = self.get_temp_dir()
        wav_dir = os.path.join(tmp_dir, "wavs")
        os.mkdir(wav_dir)
        self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
        background_dir = os.path.join(wav_dir, "_background_noise_")
        os.mkdir(background_dir)
        wav_data = self._getWavData()
        for i in range(10):
            file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
            self._saveTestWavFile(file_path, wav_data)
        model_settings = {
            "desired_samples": 160,
            "fingerprint_size": 40,
            "label_count": 4,
            "window_size_samples": 100,
            "window_stride_samples": 100,
            "dct_coefficient_count": 40,
        }
        audio_processor = input_data.AudioProcessor("", wav_dir, 10, 10, ["a", "b"],
                                                    10, 10, model_settings)
        mfcc = prepare_processing_graph()
        self.assertIsNotNone(audio_processor.wav_filename_placeholder_)
        self.assertIsNotNone(audio_processor.foreground_volume_placeholder_)
        self.assertIsNotNone(audio_processor.time_shift_padding_placeholder_)
        self.assertIsNotNone(audio_processor.time_shift_offset_placeholder_)
        self.assertIsNotNone(audio_processor.background_data_placeholder_)
        self.assertIsNotNone(audio_processor.background_volume_placeholder_)
        self.assertIsNotNone(audio_processor.mfcc_)


if __name__ == "__main__":
    test.main()
