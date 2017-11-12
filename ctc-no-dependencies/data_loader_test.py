"""

"""

import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from data_loader import DataLoader, load_wav_file, load_mfcc
from tensorflow.python.platform import test

#todo: split data into train, test, validation
#      > train_x, train_y, val_x, val_y = split_data(chars, 10, 200)
#todo: load batches(mini batches)
#todo: load mfcc and labels


class DataLoaderTest(test.TestCase):

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
        sample_data = load_wav_file(file_path)
        self.assertIsNotNone(sample_data)

    def testLoadMfcc(self):
        tmp_dir = self.get_temp_dir()
        file_path = os.path.join(tmp_dir, "load_test.wav")
        wav_data = self.getWavData()
        self.saveTestWavFile(file_path, wav_data)
        mfcc = load_mfcc(file_path)
        self.assertIsNotNone(mfcc)

    def testLoadRealMfcc(self):
        file_name = "1.wav"
        data_dir = "data"
        dir = os.path.dirname(__file__)
        file_path = os.path.join(dir, "..", data_dir, file_name)
        mfcc = load_mfcc(file_path)
        self.assertIsNotNone(mfcc)


if __name__ == "__main__":
    test.main()
