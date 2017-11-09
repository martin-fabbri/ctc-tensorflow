import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

"""
Handles loading, pre-processing, partitioning, and preparing audio for training

- dct_coefficient_count flag controls how many buckets are used for the frequency
  counting, so reducing this will shrink the input in the other dimension.

- The --window_stride_ms controls how far apart each frequency analysis sample is
  from the previous. If you increase this value, then fewer samples will be
  taken for a given duration, and the time axis of the input will shrink

- The --window_size_ms argument doesn't affect the size, but does control how wide
  the area used to calculate the frequencies is for each sample. Reducing the
  duration of the training samples, controlled by

- clip_duration_ms, can also help if the sounds you're looking for are short,
  since that also reduces the time dimension of the input. You'll need to make
  sure that all your training data contains the right audio in the initial portion
  of the clip though
"""


def prepare_processing_graph(filename):
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



# waveform = contrib_audio.decode_wav(
#  audio_binary,
#  desired_channels=1,
#  desired_samples=sample_rate,
#  name='decoded_sample_data')
#
#
# sample_rate = 16000
#
# transwav = tf.transpose(waveform[0])
#
# stfts = tf.contrib.signal.stft(transwav,
#   frame_length=2048,
#   frame_step=512,
#   fft_length=2048,
#   window_fn=functools.partial(tf.contrib.signal.hann_window,
#   periodic=False),
#   pad_end=True)
#
# spectrograms = tf.abs(stfts)
# num_spectrogram_bins = stfts.shape[-1].value
# lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0,8000.0, 128
# linear_to_mel_weight_matrix =
# tf.contrib.signal.linear_to_mel_weight_matrix(
# num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
#    upper_edge_hertz)
# mel_spectrograms = tf.tensordot(
#  spectrograms,
#  linear_to_mel_weight_matrix, 1)
# mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
# linear_to_mel_weight_matrix.shape[-1:]))
# log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
# mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
#     log_mel_spectrograms)[..., :20]