import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio


def test_me():
    return False


def load_wav_file(filename):
    """
    Loads audio file

    :arg filename
    :return: Float PCM-encoded array
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_file_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_file_placeholder)
        # todo: do we need to pass sample rate?
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        return sess.run(wav_decoder, feed_dict={wav_file_placeholder: filename}).audio.flatten()


def load_mfcc(filename):
    """
    Builds a TensorFlow graph to apply the input distortions
    :param model_settings:
    :return:
    """
    with tf.Session(graph=tf.Graph()) as sess:
        desired_samples = 16000  # todo: default should be 16K instead?
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)

        tf.transpose(wav_decoder[0])

        spectrogram = contrib_audio.audio_spectrogram(
            wav_decoder,
            window_size=1024,
            stride=512,
            magnitude_squared=True)

        mfcc = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate)

        # dct_coefficient_count=model_settings['dct_coefficient_count']

        return sess.run(mfcc, feed_dict={"wav_filename_placeholder": filename})

# def generate_mfcc(filename):
