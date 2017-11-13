"""

"""
import os
from utils import char_to_int_encode, int_to_char_decode, load_config
from tensorflow.python.platform import test

ENCODED_TARGET = [5, 16, 15, 21, 1, 2, 20, 12, 1, 14, 6, 1, 21, 16, 1, 4, 2, 19, 19, 26]
ORIGINAL_TEXT = "Dont ask me to carry"

TARGET_WITH_PUNCTUATION = [5, 16, 15, 0, 21, 1, 2, 20, 12, 1, 14, 6, 1, 21, 16, 1, 4, 2, 19, 19, 26]
TEXT_WITH_PUNCTUATION = "Don't ask me to carry."


class UtilsTest(test.TestCase):
    def testCharToIntEncode(self):
        encoded = char_to_int_encode(ORIGINAL_TEXT)
        self.assertEqual(len(encoded), len(ORIGINAL_TEXT))
        self.assertEqual(encoded, ENCODED_TARGET)

    def testIntToCharDecode(self):
        decoded = int_to_char_decode(ENCODED_TARGET)
        self.assertEqual(len(decoded), len(ENCODED_TARGET))
        self.assertEqual(decoded, ORIGINAL_TEXT.lower())

    def testEncodeTextWithPunctuation(self):
        encoded = char_to_int_encode(TEXT_WITH_PUNCTUATION)
        self.assertEqual(len(encoded), len(TEXT_WITH_PUNCTUATION.strip(".")))
        self.assertEqual(encoded, TARGET_WITH_PUNCTUATION)

    def testDecodeTargetWithAphostrophe(self):
        decoded = int_to_char_decode(TARGET_WITH_PUNCTUATION)
        self.assertEqual(len(decoded), len(TARGET_WITH_PUNCTUATION))
        self.assertEqual(decoded, TEXT_WITH_PUNCTUATION.lower().strip("."))

    def testLoadConfig(self):
        test_config_path = os.path.join(os.path.dirname(__file__), 'unitest_config.yml')
        test_config = load_config(test_config_path)
        self.assertIsNotNone(test_config)
        self.assertIsNotNone(test_config.dataset)
        self.assertIsNotNone(test_config.audio)


if __name__ == "__main__":
    test.main()