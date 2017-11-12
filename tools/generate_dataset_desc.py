"""
Generates a list of audio files that will be used for training. This script
will scan the dataset folders according to any of structures listed below.

> LibriSpeech: <data_directory>/<group>/<speaker>/[file_id1.wav, file_id2.wav, ..., speaker.trans.txt]

Fields considerations:
- audio duration -> might be helpful to short audi files during batch creation
- sample rate ?

"""

import argparse
import json
import os
import wave

def main(data_directory, output_file):
    """
    :param data_directory: dataset root folder (i.e. <data_directory>/<group>/<speaker> ...)
    :param output_file:
    :return:
    """
    labels = []
    durations = []
    keys = []
    for group in os.listdir(data_directory):
        speaker_path = os.path.join(data_directory, group)
        for speaker in os.listdir(speaker_path):
            label_file = os.path.join(
                speaker_path,
                speaker,
                f"{group}-{speaker}.trans.txt"
            )
            with open(label_file) as f:
                for line in f:
                    split = line.strip().split()
                    file_id = split[0]
                    label = ' '.join(split[1:]).lower()
                    audio_file = os.path.join(speaker_path, speaker, file_id) + '.wav'
                    audio = wave.open(audio_file)
                    duration = float(audio.getnframes()) / audio.getframerate()
                    audio.close()
                    keys.append(audio_file)
                    durations.append(duration)
                    labels.append(label)

        with open(output_file, 'w') as out_file:
            for key, duration, label in zip(keys, durations, labels):
                line = json.dumps({'key': key, 'duration': duration, 'text': label})
                out_file.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, help="Path to dataset root directory")
    parser.add_argument('output_file', type=str, help="Path to output file")
    args = parser.parse_args()
    main(args.data_directory, args.output_file)