"""

"""
import os
import namedtupled
import yaml
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.platform import test
from loaders.data_loader import DataLoader, _load_metadata_from_desc_file


#todo: split data into train, test, validation
#      > train_x, train_y, val_x, val_y = split_data(chars, 10, 200)
#todo: load batches(mini batches)
#todo: load mfcc and labels


class DataLoaderTest(test.TestCase):
    def get_test_config(self, desc_file_path):
        return {
            "dataset": {
                "path": desc_file_path,
                "max_duration_seg": 10.0,
                "test_size": 0.2
            },
            "train": {
                "minibatch_size": 2,
                "shuffle": False
            },
            "audio": {
                "window_size_ms": 550,
                "window_stride_ms": 350,
                "dct_coefficient_count": 13
            }
        }

    def getWavData(self):
        with self.test_session() as sess:
            sample_data = tf.zeros([1000, 2])
            wav_encoder = contrib_audio.encode_wav(sample_data, 16000)
            wav_data = sess.run(wav_encoder)
        return wav_data

    def saveTestWavFiles(self, file_paths):
        wav_data = self.getWavData()
        for fp in file_paths:
            with open(fp, "wb") as f:
                f.write(wav_data)

    def saveDescMetadataFile(self, file_path, keys):
        with open(file_path, "w") as out_file:
            data = [{"key": key, "duration": 1.0, "text": f"Dummy text {key.split('/')[-1].replace('.', '')}"}
                    for key in keys]
            formatted_data = yaml.dump(data)
            out_file.write(formatted_data + '\n')

    def testLoadMetadataFromFile(self):
        dir = os.path.dirname(__file__)
        file_path = os.path.join(dir, "unittest_dataset_specs.yml")
        audio_path, texts = _load_metadata_from_desc_file(file_path, 10.0)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(texts)

    def testDataLoadSplit(self):
        tmp_dir = self.get_temp_dir()
        wav_dir = os.path.join(tmp_dir, "wavs")
        os.mkdir(wav_dir)
        test_config = self.get_test_config(os.path.join(wav_dir, "test_spec.yml"))
        config = namedtupled.map(test_config)
        keys = [os.path.join(wav_dir, f"{i}.wav") for i in ['a', 'b', 'c', 'd', 'f']]
        self.saveDescMetadataFile(config.dataset.path, keys)
        self.saveTestWavFiles(keys)
        data_loader = DataLoader(config)
        self.assertIsNotNone(data_loader.dataset)
        self.assertIsNotNone(data_loader.audio)
        self.assertIsNotNone(data_loader.train)
        self.assertIsNotNone(data_loader.audio_train)
        self.assertIsNotNone(data_loader.audio_test)
        self.assertIsNotNone(data_loader.text_train)
        self.assertIsNotNone(data_loader.text_test)
        self.assertEqual(len(data_loader.audio_train), 4)
        self.assertEqual(len(data_loader.text_train), 4)
        self.assertEqual(len(data_loader.audio_test), 1)
        self.assertEqual(len(data_loader.text_test), 1)

    def testNextTrainingBatch(self):
        tmp_dir = self.get_temp_dir()
        wav_dir = os.path.join(tmp_dir, "wavs")
        os.mkdir(wav_dir)
        test_config = self.get_test_config(os.path.join(wav_dir, "test_spec.yml"))
        config = namedtupled.map(test_config)
        keys = [os.path.join(wav_dir, f"{i}.wav") for i in ['a', 'b', 'c', 'd', 'f']]
        self.saveDescMetadataFile(config.dataset.path, keys)
        self.saveTestWavFiles(keys)
        data_loader = DataLoader(config)
        with self.test_session() as sess:
            # test a single iteration
            features, labels, texts = next(data_loader.iterate_training_batch(sess))
            self.assertIsNotNone(features)
            self.assertEquals(len(features), 2)
            self.assertEquals(len(labels), 2)
            self.assertEquals(len(texts), 2)
            # test all iterations
            num_tuples_per_iteration = []
            for features, labels, texts in data_loader.iterate_training_batch(sess):
                num_tuples_per_iteration.append(len(features))
            self.assertEquals(len(num_tuples_per_iteration), 2)
            self.assertEquals(num_tuples_per_iteration[0], 2)


if __name__ == "__main__":
    test.main()
