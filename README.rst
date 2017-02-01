Text Autoencoder
================

This is an implementation of a recurrent neural network that reads an input text, encodes it in its memory cell, and then reconstructs the inputs. This is basically the idea presented by `Sutskever et al. (2014) <https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf>`_

**Why?** The point of training an autoencoder is to make an RNN learn how to compress a relatively long sequence into a limited, dense vector. Once we have a fixed-size representation of a sentence, there's a lot we can do with it.

We can work with single sentences (classifying them with respect to sentiment, topic, authorship, etc), or more than one at a time (checking for similarities, contradiction, question/answer pairs, etc.) Another successful application is to encode one sentence in one language and use a different autoencoder to decode it into another language, e.g. `Cho et al. (2014) <https://arxiv.org/abs/1406.1078>`_.

Implementation
--------------

The autoencoder is implemented with `Tensorflow <http://tensorflow.org>`_. Specifically, it uses a bidirectional LSTM (but it can be configured to use a simple LSTM instead).

In the encoder step, the LSTM reads the whole input sequence; its outputs at each time step are ignored.

Then, in the decoder step, a special symbol *GO* is read, and the output of the LSTM is fed to a linear layer with the size of the vocabulary. The chosen word (i.e., the one with the highest score) is the next input to the decoder. This goes on until a special symbol *EOS* is produced.

The weights of the encoder and decoder are shared.

Performance notes
^^^^^^^^^^^^^^^^^

- Even for small vocabularies (a few thousand words), training the network over all possible outputs at each time step is very expensive computationally. Instead, we just sample the weights of 100 possible words. During inference time, there is no way around it, but the computational cost is much lesser.

- For better decoder performance, a beam search is preferable to the currently used greedy choice.

Scripts
-------

* ``prepare-data.py``: reads a text file and create numpy files that can be used to train an autoencoder

* ``train-autoencoder.py``: train a new autoencoder model

* ``interactive.py``: run a trained autoencoder that reads input from stdin. It can be fun to test the boundaries of your trained model :)

* ``codify-sentences.py``: run the encoder part of a trained autoencoder on sentences read from a text file. The encoded representation is saved as a numpy file

You can run any of the scripts with ``-h`` to get information about what arguments they accept.

