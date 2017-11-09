import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
def load_wav_file(filename):
    """
    Loads audio file

    :arg filename
    :return: Float PCM-encoded array
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_file_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_file_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        return sess.run(wav_decoder, feed_dict={wav_file_placeholder: filename}).audio.flatten()


# waveform = contrib_audio.decode_wav(
#  audio_binary,
#  desired_channels=1,
#  desired_samples=sample_rate,
#  name='decoded_sample_data')