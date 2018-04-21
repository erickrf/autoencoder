# -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals

import argparse
import os
import numpy as np
import math
from collections import defaultdict, Counter

"""
This script processes an input text file to produce data in binary
format to be used with the autoencoder (binary is much faster to read).
"""


def load_data_memory_friendly(path, max_size, min_occurrences=10,
                              valid_proportion=0.01):
    """
    Return a tuple (dict, dict, list) where each dict maps names to
    sentence matrices and sizes arrays (first is train, second is validation);
    the list is the vocabulary
    """
    token_counter = Counter()
    size_counter = Counter()

    # first pass to build vocabulary and count sentence sizes
    print('Creating vocabulary...')
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            tokens = line.split()
            sent_size = len(tokens)
            if sent_size > max_size:
                continue

            # keep track of different size bins, with bins for
            # 1-10, 11-20, 21-30, etc
            top_bin = int(math.ceil(sent_size / 10) * 10)
            size_counter[top_bin] += 1
            token_counter.update(tokens)

    # sort it keeping the order
    vocabulary = [w for w, count in token_counter.most_common()
                  if count >= min_occurrences]
    # this might break the ordering, but hopefully is not a problem
    vocabulary.insert(0, '</s>')
    vocabulary.insert(1, '<unk>')
    mapping = zip(vocabulary, range(len(vocabulary)))
    dd = defaultdict(lambda: 1, mapping)

    # now read the corpus again to fill the sentence matrix
    print('Converting word to indices...')
    train_data = {}  # dictionary to be used with numpy.savez
    valid_data = {}
    for threshold in size_counter:
        min_threshold = threshold - 9
        num_sentences = size_counter[threshold]
        print('Converting %d sentences with length between %d and %d'
              % (num_sentences, min_threshold, threshold))
        sents, sizes = create_sentence_matrix(path, num_sentences,
                                              min_threshold, threshold, dd)

        # shuffle sentences and sizes with the sime RNG state
        state = np.random.get_state()
        np.random.shuffle(sents)
        np.random.set_state(state)
        np.random.shuffle(sizes)

        ind = int(len(sents) * valid_proportion)
        valid_sentences = sents[:ind]
        valid_sizes = sizes[:ind]
        train_sentences = sents[ind:]
        train_sizes = sizes[ind:]

        train_data['sentences-%d' % threshold] = train_sentences
        train_data['sizes-%d' % threshold] = train_sizes
        valid_data['sentences-%d' % threshold] = valid_sentences
        valid_data['sizes-%d' % threshold] = valid_sizes

    print('Numeric representation ready')
    return train_data, valid_data, vocabulary


def create_sentence_matrix(path, num_sentences, min_size,
                           max_size, word_dict):
    """
    Create a sentence matrix from the file in the given path.
    :param path: path to text file
    :param min_size: minimum sentence length, inclusive
    :param max_size: maximum sentence length, inclusive
    :param num_sentences: number of sentences expected
    :param word_dict: mapping of words to indices
    :return: tuple (2-d matrix, 1-d array) with sentences and
        sizes
    """
    sentence_matrix = np.full((num_sentences, max_size), 0, np.int32)
    sizes = np.empty(num_sentences, np.int32)
    i = 0
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            tokens = line.split()
            sent_size = len(tokens)
            if sent_size < min_size or sent_size > max_size:
                continue

            array = np.array([word_dict[token] for token in tokens])
            sentence_matrix[i, :sent_size] = array
            sizes[i] = sent_size
            i += 1

    return sentence_matrix, sizes


def load_data(path, max_size, min_occurrences=10):
    """
    Load data from a text file and creates the numpy arrays
    used by the autoencoder.

    :return: a tuple (sentences, sizes, vocabulary).
        sentences is a 2-d matrix padded with EOS
        sizes is a 1-d array with each sentence size
        vocabulary is a list of words positioned according to their indices
    """
    sentences = []
    sizes = []
    longest_sent_size = 0
    index = [0]  # hack -- use a mutable object to be
                 # accessed inside the nested function
                 # at first, 0 means padding/EOS

    def on_new_word():
        index[0] += 1
        return index[0]
    word_dict = defaultdict(on_new_word)

    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            tokens = line.split()
            sent_size = len(tokens)
            if sent_size > max_size:
                continue
            sentences.append([word_dict[token]
                             for token in tokens])
            sizes.append(sent_size)
            if sent_size > longest_sent_size:
                longest_sent_size = sent_size

    reverse_word_dict = {v: k for k, v in word_dict.items()}
    reverse_word_dict[0] = '</s>'
    # we initialize the matrix now that we know the number of sentences
    sentence_matrix = np.full((len(sentences), longest_sent_size),
                              0, np.int32)

    for i, sent in enumerate(sentences):
        sentence_array = np.array(sent)
        sentence_matrix[i, :sizes[i]] = sentence_array

    # count occurrences of tokens on the remaining sentences
    # counter: index -> num_occurences
    counter = Counter(sentence_matrix.flat)

    # 0 signs the EOS token, it should be counted once per sentence
    counter[0] = len(sentence_matrix)

    # these words will be replaced by the unk token
    unk_words = [(w, counter[w]) for w in counter
                 if counter[w] < min_occurrences]
    unk_count = sum(item[1] for item in unk_words)
    unk_index = len(counter)   # make the unknown index the last one
    counter[unk_index] = unk_count
    reverse_word_dict[unk_index] = '<unk>'

    # now we sort word indices by frequency (this works better with some
    # sampling techniques such as Noise Constrastive Estimation)
    replacements = {}
    word_list = []
    for new_index, (old_index, count) in enumerate(counter.most_common()):
        if count < min_occurrences:
            # we can break the loop because the next ones
            # have equal or lower counts
            break

        replacements[old_index] = new_index
        word_list.append(reverse_word_dict[old_index])

    new_unk_index = replacements[unk_index]
    replacements_with_unk = defaultdict(lambda: new_unk_index,
                                        replacements)
    original_shape = sentence_matrix.shape
    replaced = np.array([replacements_with_unk[w]
                         for w in sentence_matrix.flat],
                        dtype=np.int32)
    sentence_matrix = replaced.reshape(original_shape)

    sizes_array = np.array(sizes, dtype=np.int32)
    return sentence_matrix, sizes_array, word_list


def write_vocabulary(words, path):
    """
    Write the contents of word_dict to the given path.
    """
    text = '\n'.join(words)
    with open(path, 'wb') as f:
        f.write(text.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Text file previously tokenized '
                                      '(by whitespace) and preprocessed')
    parser.add_argument('output', help='Directory to save the data')
    parser.add_argument('--max-length',
                        help='Maximum sentence size (default 60)',
                        type=int, default=60, dest='max_length')
    parser.add_argument('--min-freq', help='Minimum times a word must '
                                           'occur (default 10)',
                        default=10, type=int, dest='min_freq')
    parser.add_argument('--valid', type=float, default=0.01,
                        dest='valid_proportion',
                        help='Proportion of the validation dataset '
                             '(default 0.01)')
    args = parser.parse_args()

    train_data, valid_data, words = load_data_memory_friendly(
        args.input, args.max_length, args.min_freq, args.valid_proportion)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    path = os.path.join(args.output, 'valid-data.npz')
    np.savez(path, **valid_data)

    path = os.path.join(args.output, 'train-data.npz')
    np.savez(path, **train_data)

    path = os.path.join(args.output, 'vocabulary.txt')
    write_vocabulary(words, path)
