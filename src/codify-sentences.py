# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
import numpy as np
import tensorflow as tf

import utils
from autoencoder import TextAutoencoder

"""
Run the encoder part of the autoencoder in a corpus to generate
the memory cell representation for them.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory with saved model')
    parser.add_argument('input', help='File with one sentence per line')
    parser.add_argument('vocabulary', help='File with autoencoder vocabulary')
    parser.add_argument('output', help='Numpy file to write output')
    args = parser.parse_args()

    wd = utils.WordDictionary(args.vocabulary)
    sentences, sizes = utils.load_text_data(args.input, wd)
    sess = tf.InteractiveSession()
    model = TextAutoencoder.load(args.model, sess)

    # feed blocks of 5k sentences
    num_sents = 5000
    next_index = 0
    all_states = []
    while next_index < len(sentences):
        batch = sentences[next_index:next_index + num_sents]
        batch_sizes = sizes[next_index:next_index + num_sents]
        next_index += num_sents
        state = model.encode(sess, batch, batch_sizes)
        all_states.append(state)

    state = np.vstack(all_states)
    np.save(args.output, state)
