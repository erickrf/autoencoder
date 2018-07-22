# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

"""
Interactive evaluation for the RTE networks.
"""

import argparse
import logging
import tensorflow as tf
import numpy as np
from six.moves import input

import utils
import autoencoder


class SentenceWrapper(object):
    """
    Class for the basic sentence preprocessing needed to make it readable
    by the networks.
    """
    def __init__(self, sentence, word_dict, lower):
        if lower:
            sentence = sentence.lower()
        self.tokens = sentence.split()
        self.indices = np.array([word_dict[token]
                                 for token in self.tokens])

    def __len__(self):
        return len(self.tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory with saved model files')
    parser.add_argument('vocabulary', help='Vocabulary file')
    parser.add_argument('-l', dest='lower', action='store_true',
                        help='Convert text to lowercase')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info('Reading model')
    sess = tf.InteractiveSession()
    model = autoencoder.TextAutoencoder.load(args.model, sess)
    word_dict = utils.WordDictionary(args.vocabulary)
    index_dict = word_dict.inverse_dictionary()

    while True:
        string = input('Type tokenized sentence: ')
        sent = SentenceWrapper(string, word_dict, args.lower)
        answer = model.run(sess, [sent.indices], [len(sent)])
        answer_words = [index_dict[i] for i in answer]
        answer_str = ' '.join(answer_words)
        print('Model output:', answer_str)
