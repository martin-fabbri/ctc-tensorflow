"""
todo: Document
"""
import os
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


def load_config(file_path, append_path=None):
    with open(file_path, encoding='utf-8-sig') as ymal_file:
        props = yaml.load(ymal_file)
        if append_path is not None:
            props["dataset"]["path"] = os.path.join(append_path, props["dataset"]["path"])
        return namedtupled.map(props)
