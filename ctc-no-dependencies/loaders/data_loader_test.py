"""

"""

import os

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.platform import test

from loaders.data_loader import DataLoader


#todo: split data into train, test, validation
#      > train_x, train_y, val_x, val_y = split_data(chars, 10, 200)
#todo: load batches(mini batches)
#todo: load mfcc and labels


class DataLoaderTest(test.TestCase):
    def setUp(self):
        self.data_loader = DataLoader()

    def getWavData(self):
        with self.test_session() as sess:
            sample_data = tf.zeros([1000, 2])
            wav_encoder = contrib_audio.encode_wav(sample_data, 16000)
            wav_data = sess.run(wav_encoder)
        return wav_data

    def saveTestWavFile(self, filename, wav_data):
        with open(filename, "wb") as f:
            f.write(wav_data)

    def testDescriptorFileEmpty(self):
        with self.assertRaises(Exception) as e:
            _ = DataLoader("")
        self.assertTrue("Descriptor file not found." in str(e.exception))

    def testLoadWavFile(self):
        tmp_dir = self.get_temp_dir()
        file_path = os.path.join(tmp_dir, "load_test.wav")
        wav_data = self.getWavData()
        self.saveTestWavFile(file_path, wav_data)
        sample_data = self.data_loader.load_wav_file(file_path)
        self.assertIsNotNone(sample_data)

    def testLoadMfcc(self):
        tmp_dir = self.get_temp_dir()
        file_path = os.path.join(tmp_dir, "load_test.wav")
        wav_data = self.getWavData()
        self.saveTestWavFile(file_path, wav_data)
        mfcc = self.data_loader.load_mfcc(file_path)
        self.assertIsNotNone(mfcc)

    def testLoadRealMfcc(self):
        file_name = "1.wav"
        data_dir = "data"
        dir = os.path.dirname(__file__)
        file_path = os.path.join(dir, "..", data_dir, file_name)
        mfcc = self.data_loader.load_mfcc(file_path)
        self.assertIsNotNone(mfcc)

    # def testPrepareProcessingGraph(self):
    #     tmp_dir = self.get_temp_dir()
    #     wav_dir = os.path.join(tmp_dir, "wavs")
    #     os.mkdir(wav_dir)
    #     self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    #     background_dir = os.path.join(wav_dir, "_background_noise_")
    #     os.mkdir(background_dir)
    #     wav_data = self._getWavData()
    #     for i in range(10):
    #         file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
    #         self._saveTestWavFile(file_path, wav_data)
    #     model_settings = {
    #         "desired_samples": 160,
    #         "fingerprint_size": 40,
    #         "label_count": 4,
    #         "window_size_samples": 100,
    #         "window_stride_samples": 100,
    #         "dct_coefficient_count": 40,
    #     }
    #     audio_processor = input_data.AudioProcessor("", wav_dir, 10, 10, ["a", "b"],
    #                                                 10, 10, model_settings)
    #     self.assertIsNotNone(audio_processor.wav_filename_placeholder_)
    #     self.assertIsNotNone(audio_processor.foreground_volume_placeholder_)
    #     self.assertIsNotNone(audio_processor.time_shift_padding_placeholder_)
    #     self.assertIsNotNone(audio_processor.time_shift_offset_placeholder_)
    #     self.assertIsNotNone(audio_processor.background_data_placeholder_)
    #     self.assertIsNotNone(audio_processor.background_volume_placeholder_)
    #     self.assertIsNotNone(audio_processor.mfcc_)


if __name__ == "__main__":
    test.main()
