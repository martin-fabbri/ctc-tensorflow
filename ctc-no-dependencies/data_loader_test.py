"""

"""

import os
import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from data_loader import DataLoader
from tensorflow.python.platform import test

#todo: split data into train, test, validation
#      > train_x, train_y, val_x, val_y = split_data(chars, 10, 200)
#todo: load batches(mini batches)
#todo: load mfcc and labels

class DataLoaderTest(test.TestCase):
    def testDummy(self):
        loader = DataLoader("test.json", window_ms=25, max_freq=8000, max_duration=899999)

        print(len(loader.audio_train))
        print(len(loader.audio_test))

        for val in loader.next_training_batch(minibatch_size=16, sort_by_duration=False, shuffle=True):
            print(val)

        self.assertFalse(False)


if __name__ == "__main__":
    test.main()
