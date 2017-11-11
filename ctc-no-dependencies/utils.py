"""
todo: Document
"""
import string
from mappings import char_map, index_map
from constants import PUNCTUATION

def char_to_int_encode(text):
    """
    Use a character map and convert to an integer sequence
    :param text:
    :return:
    """
    # # Adding blank label
    # targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
    #
    # # Transform char into index
    # targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
    #                       for x in targets])

    return [char_map["<SPACE>"] if char == " " else char_map[char] for char in text.lower()
            if char not in PUNCTUATION]


def int_to_char_decode(encoded):
    """
    Decode a prediction simply converting an integer sequence to symbols
    :param encoded: [int] predicted array of encoded symbols
    :return:
    """
    return "".join([index_map[i] for i in encoded])
