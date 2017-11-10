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
        wav_decoder = contrib_audio.decode_wav(
            wav_loader,
            desired_channels=1)
        return sess.run(wav_decoder, feed_dict={wav_file_placeholder: filename}).audio.flatten()


def load_mfcc(filename):
    """
    Builds a TensorFlow graph to apply the input distortions
      Args:

    sample_rate: How many samples per second are in the input audio files.
    ? clip_duration_ms: How many samples to analyze for the audio pattern.
    ? clip_stride_ms: How often to run recognition. Useful for models with cache.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.

    :param

    :return:
    """
    with tf.Session(graph=tf.Graph()) as sess:
        desired_samples = 16000  # todo: default should be 16K instead?
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(
            wav_loader,
            desired_channels=1)

        # window_size_ms=25 (recommended in literature for speech recognition) ->
        #   > How wide the input window is in samples. For the highest
        #     efficiency this should be a power of two,
        #     but other values are accepted.
        #   > python_speech_features also defaults window_size to 25

        #
        # window_stride_ms=10 (recommended in literature for speech recognition) ->
        #   > How widely apart the center of adjacent sample windows
        #     should be.
        #   > python_speech_features also defaults winstep to 10
        #
        # magnitude_squared=False (Default) -> todo: should we chanfe this parameter to True?
        #   > Whether to return the squared magnitude or just the
        #     magnitude. Using squared magnitude can avoid extra
        #     calculations.
        #
        # dct_coefficient_count=13 (default) ->
        #   > How many output channels to produce per time slice.
        #   > python_speech_features' numcep default is 13

        spectrogram = contrib_audio.audio_spectrogram(
            wav_decoder.audio,
            window_size=550,  # for efficiency make it a power of 2
            stride=350,
            #magnitude_squared=True
        )

        mfcc = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=13
        )
        return sess.run(mfcc,
                        feed_dict={wav_filename_placeholder: filename})

# def generate_mfcc(filename):
