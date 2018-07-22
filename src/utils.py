# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

"""
Utilities for working with the autoencoder.
"""

from collections import defaultdict
import numpy as np


class WordDictionary(object):
    """
    Simple class to wrap a defaultdict and keep track of the vocabulary
    size and indices for unknown and padding.
    """
    def __init__(self, path):
        words = read_word_list(path)
        self.vocabulary_size = len(words)

        # If there are repeated words, they will be mapped to the same index
        # If there were errors reading the file (for decoding), they will be
        # represented as an empty string and collapsed together
        # In both cases, the order of words is preserved.
        index_range = range(len(words))
        mapping = zip(words, index_range)
        self.oov_index = words.index('<unk>')
        self.d = defaultdict(lambda: self.oov_index, mapping)
        self.eos_index = self.d['</s>']

    def __getitem__(self, item):
        return self.d[item]

    def __contains__(self, item):
        return item in self.d

    def __len__(self):
        return len(self.d)

    def inverse_dictionary(self):
        """
        Return a dictionary mapping indices to words
        """
        return {v: k for (k, v) in self.d.items()}


class Dataset(object):
    """
    Class to manage an autoencoder dataset. It contains a sentence
    matrix, an array with their sizes and functions to facilitate access.
    """
    def __init__(self, sentences, sizes):
        """
        :param sentences: either a matrix or a list of matrices
            (which could have different shapes)
        :param sizes: either an array or a list of arrays
            (which could have different shapes)
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
            sizes = [sizes]

        self.sentence_matrices = sentences
        self.sizes = sizes
        self.num_items = sum(len(array) for array in sizes)
        self.next_batch_ind = 0
        self.last_matrix_ind = 0
        self.epoch_counter = 0
        self.largest_len = max(sent.shape[1] for sent in sentences)

    def __len__(self):
        return self.num_items

    def reset_epoch_counter(self):
        self.epoch_counter = 0

    def next_batch(self, batch_size):
        """
        Return the next batch (keeping track of the last, or from the beginning
        if this is the first call).

        Sentences are grouped in batches according to their sizes (similar sizes
        go together).

        :param batch_size: number of items to return
        :return: a tuple (sentences, sizes) with at most `batch_size`
            items. If there are not enough `batch_size`, return as much
            as there are
        """
        matrix = self.sentence_matrices[self.last_matrix_ind]
        if self.next_batch_ind >= len(matrix):
            self.last_matrix_ind += 1
            if self.last_matrix_ind >= len(self.sentence_matrices):
                self.epoch_counter += 1
                self.last_matrix_ind = 0

            self.next_batch_ind = 0
            matrix = self.sentence_matrices[self.last_matrix_ind]

        sizes = self.sizes[self.last_matrix_ind]
        from_ind = self.next_batch_ind
        to_ind = self.next_batch_ind + batch_size
        batch_sentences = matrix[from_ind:to_ind]
        batch_sizes = sizes[from_ind:to_ind]
        self.next_batch_ind = to_ind

        return batch_sentences, batch_sizes

    def join_all(self, eos, max_size=None, shuffle=True):
        """
        Join all sentence matrices and return them.

        :param eos: the eos index to fill smaller matrices
        :param max_size: number of columns in the resulting matrix
        :param shuffle: whether to shuffle data before returning
        :return: (sentences, sizes)
        """
        if max_size is None:
            max_size = max(matrix.shape[1]
                           for matrix in self.sentence_matrices)
        padded_matrices = []
        for matrix in self.sentence_matrices:
            if matrix.shape[1] == max_size:
                padded = matrix
            else:
                diff = max_size - matrix.shape[1]
                padded = np.pad(matrix, [(0, 0), (0, diff)],
                                'constant', constant_values=eos)
            padded_matrices.append(padded)

        sentences = np.vstack(padded_matrices)
        sizes = np.hstack(self.sizes)
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(sentences)
            np.random.set_state(state)
            np.random.shuffle(sizes)
        return sentences, sizes


def read_word_list(path):
    """
    Read the contents of a file and return a list of lines
    """
    words = []

    with open(path, 'rb') as f:
        # try:
        for line in f:
            line = line.decode('utf-8', 'ignore')
            line = line.strip()
            words.append(line)
        # except UnicodeDecodeError:
        #     print('Error trying to decode word, skipping:', line)

    return words


def load_binary_data(path):
    """
    Load a numpy archive. It can have either a single 'sentences'
    and a single 'sizes' or many 'sentences-x' and 'sizes-x'.
    """
    data = np.load(path)
    if 'sentences' in data:
        return Dataset(data['sentences'], data['sizes'])

    sent_names = sorted(name for name in data.files
                        if name.startswith('sentences'))
    size_names = sorted(name for name in data.files
                        if name.startswith('sizes'))
    sents = []
    sizes = []

    for sent_name, size_name in zip(sent_names, size_names):
        sents.append(data[sent_name])
        sizes.append(data[size_name])

    return Dataset(sents, sizes)


def load_text_data(path, word_dict):
    """
    Read the given path, which should have one sentence per line

    :param path: path to file
    :param word_dict: dictionary mapping words to embedding
        indices
    :type word_dict: WordDictionary
    :return: a tuple with a matrix of sentences and an array
        of sizes
    """
    max_len = 0
    all_indices = []
    sizes = []
    with open(path, 'rb') as f:
        for line in f:
            tokens = line.decode('utf-8').split()
            this_len = len(tokens)
            if this_len > max_len:
                max_len = this_len
            sizes.append(this_len)

            inds = [word_dict[token] for token in tokens]
            all_indices.append(inds)

    shape = (len(all_indices), max_len)
    sizes = np.array(sizes)
    matrix = np.full(shape, word_dict.eos_index, np.int32)
    for i, inds in enumerate(all_indices):
        matrix[i, :len(inds)] = inds

    return matrix, sizes
