import os
import time

import numpy as np
import tensorflow as tf

from loaders import DataLoader
from utils import char_map, int_to_char_decode, load_config


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape


# Some configs
num_features = 13
num_classes = len(char_map)

# training hyper-parameters
num_epochs = 200
num_hidden = 50
num_layers = 1  # lstm layers
batch_size = 1  # ????
learning_rate = 0.01
momentum = 0.9
initial_learning_rate = 1e-2

# dataset
num_examples = 1
num_batches_per_epoch = int(num_examples/batch_size)

# loading audio processing config
current_dir = os.path.dirname(__file__)
config_path = os.path.join(current_dir, "one_clip_config.yml")
config = load_config(config_path, current_dir)
data_loader = DataLoader(config)


# e.g: log filter bank or MFCC features
# Has size [batch_size, max_stepsize, num_features], but the
# batch_size and max_stepsize can vary along each step
inputs = tf.placeholder(tf.float32, [None, None, num_features])

# Here we use sparse_placeholder that will generate a
# SparseTensor required by ctc_loss op.
targets = tf.sparse_placeholder(tf.int32)

# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None])

# Defining the cell
# Can be:
#   tf.nn.rnn_cell.RNNCell
#   tf.nn.rnn_cell.GRUCell
cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

# Stacking rnn cells
stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                    state_is_tuple=True)

# The second output is the last state and we will no use that
outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

shape = tf.shape(inputs)
batch_s, max_timesteps = shape[0], shape[1]

# Reshaping to apply the same weights over the timesteps
outputs = tf.reshape(outputs, [-1, num_hidden])

# Truncated normal with mean 0 and stdev=0.1
# Tip: Try another initialization
# see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
W = tf.Variable(tf.truncated_normal([num_hidden,
                                     num_classes],
                                    stddev=0.1))
# Zero initialization
# Tip: Is tf.zeros_initializer the same?
b = tf.Variable(tf.constant(0., shape=[num_classes]))

# Doing the affine projection
logits = tf.matmul(outputs, W) + b

# Reshaping back to the original shape
logits = tf.reshape(logits, [batch_s, -1, num_classes])

# Time major
logits = tf.transpose(logits, (1, 0, 2))

loss = tf.nn.ctc_loss(targets, logits, seq_len)
cost = tf.reduce_mean(loss)

optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                       0.9).minimize(cost)

# Option 2: tf.nn.ctc_beam_search_decoder
# (it's slower but you'll get better results)
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

# Inaccuracy: label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                      targets))

with tf.Session() as session:
    features, labels, texts = next(data_loader.iterate_training_batch(session, append_path=current_dir))

    # Tranform in 3D array
    train_inputs = features[0]
    train_seq_len = [train_inputs.shape[1]]

    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from([labels[0]])

    # We don't have a validation dataset :(
    val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len

    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(num_batches_per_epoch):
            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len}
            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    seq_len: val_seq_len}

        val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))
    # Decoding
    d = session.run(decoded[0], feed_dict=feed)
    str_decoded = int_to_char_decode(d.values)

    print('Original:\n%s' % texts[0])
    print('Decoded:\n%s' % str_decoded)