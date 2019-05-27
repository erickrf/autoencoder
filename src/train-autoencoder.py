# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
Script for training the autoencoder.
"""

import tensorflow as tf
import logging
import numpy as np
import argparse

import utils
import autoencoder


def show_parameter_count(variables):
    """
    Count and print how many parameters there are.
    """
    total_parameters = 0
    for variable in variables:
        name = variable.name

        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        print('{}: {} ({} parameters)'.format(name,
                                              shape,
                                              variable_parametes))
        total_parameters += variable_parametes

    print('Total: {} parameters'.format(total_parameters))


def load_or_create_embeddings(path, vocab_size, embedding_size):
    """
    If path is given, load an embeddings file. If not, create a random
    embedding matrix with shape (vocab_size, embedding_size)
    """
    if path is not None:
        return np.load(path)

    embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_size))
    return embeddings.astype(np.float32)


if __name__ == '__main__':
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)
    parser.add_argument('save_dir', help='Directory to file to save trained '
                                         'model')
    parser.add_argument('-n', help='Embedding size (if embeddings not given)',
                        default=300, dest='embedding_size', type=int)
    parser.add_argument('-u', help='Number of LSTM units (when using a '
                                   'bidirectional model, this is doubled in '
                                   'practice)', default=500,
                        dest='lstm_units', type=int)
    parser.add_argument('-r', help='Initial learning rate', default=0.001,
                        dest='learning_rate', type=float)
    parser.add_argument('-b', help='Batch size', default=32,
                        dest='batch_size', type=int)
    parser.add_argument('-e', help='Number of epochs', default=1,
                        dest='num_epochs', type=int)
    parser.add_argument('-d', help='Dropout keep probability', type=float,
                        dest='dropout_keep', default=1.0)
    parser.add_argument('-i',
                        help='Number of batches between performance report',
                        dest='interval', type=int, default=1000)
    parser.add_argument('--mono', help='Use a monodirectional LSTM '
                                       '(bidirectional is used by default)',
                        action='store_false', dest='bidirectional')
    parser.add_argument('--te', help='Train embeddings. If not given, they are '
                                     'frozen. (always true if embeddings are '
                                     'not given)',
                        action='store_true', dest='train_embeddings')
    parser.add_argument('--embeddings',
                        help='Numpy embeddings file. If not supplied, '
                             'random embeddings are generated.')
    parser.add_argument('vocab', help='Vocabulary file')
    parser.add_argument('train', help='Training set')
    parser.add_argument('valid', help='Validation set')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    sess = tf.Session()
    wd = utils.WordDictionary(args.vocab)
    embeddings = load_or_create_embeddings(args.embeddings, wd.vocabulary_size,
                                           args.embedding_size)

    logging.info('Reading training data')
    train_data = utils.load_binary_data(args.train)
    logging.info('Reading validation data')
    valid_data = utils.load_binary_data(args.valid)
    logging.info('Creating model')

    train_embeddings = args.train_embeddings if args.embeddings else True
    model = autoencoder.TextAutoencoder(args.lstm_units,
                                        embeddings, wd.eos_index,
                                        train_embeddings=train_embeddings,
                                        bidirectional=args.bidirectional)

    sess.run(tf.global_variables_initializer())
    show_parameter_count(model.get_trainable_variables())
    logging.info('Initialized the model and all variables. Starting training.')
    model.train(sess, args.save_dir, train_data, valid_data, args.batch_size,
                args.num_epochs, args.learning_rate,
                args.dropout_keep, 5.0, report_interval=args.interval)
    logging.info('Finished training after '+str(args.num_epochs)+' epochs.')
