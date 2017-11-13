"""
todo: Document
"""
import yaml
import namedtupled
from utils.constants import PUNCTUATION
from utils.mappings import char_map, index_map


def char_to_int_encode(text):
    """
    Use a character map and convert to an integer sequence
    :param text:
    :return:
    """
    return [char_map["<SPACE>"] if char == " " else char_map[char] for char in text.lower()
            if char not in PUNCTUATION]


def int_to_char_decode(encoded):
    """
    Decode a prediction simply converting an integer sequence to symbols
    :param encoded: [int] predicted array of encoded symbols
    :return:
    """
    return "".join([index_map[i] for i in encoded])


def load_config(absolute_json_file_path):
    with open(absolute_json_file_path, encoding='utf-8-sig') as ymal_file:
        return namedtupled.map(yaml.load(ymal_file))
