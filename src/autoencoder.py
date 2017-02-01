# -*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf
import numpy as np
import logging
import json
import os


class TextAutoencoder(object):
    """
    Class that encapsulates the encoder-decoder architecture to
    reconstruct pieces of text.
    """

    def __init__(self, lstm_units, num_time_steps, embeddings,
                 eos, train=True, train_embeddings=False,
                 bidirectional=True):
        """
        Initialize the encoder/decoder and creates Tensor objects

        :param lstm_units: number of LSTM units
        :param num_time_steps: maximum number of time steps, i.e., token
            (when using a trained model, this can be None)
        :param embeddings: numpy array with initial embeddings
        :param eos: index of the EOS symbol in the embedding matrix
        :param train_embeddings: whether to adjust embeddings during training
        :param bidirectional: whether to create a bidirectional autoencoder
            (if False, a simple linear LSTM is used)
        """
        self.eos = eos
        self.bidirectional = bidirectional
        self.vocab_size = embeddings.shape[0]
        self.embedding_size = embeddings.shape[1]
        self.num_time_steps = num_time_steps
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # the sentence is the object to be memorized
        self.sentence = tf.placeholder(tf.int32,
                                       [None, num_time_steps],
                                       'sentence')
        self.sentence_size = tf.placeholder(tf.int32, [None],
                                            'sentence_size')
        self.l2_constant = tf.placeholder(tf.float32, name='l2_constant')
        self.clip_value = tf.placeholder(tf.float32, name='clip')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')

        self.decoder_step_input = tf.placeholder(tf.int32,
                                                 [None],
                                                 'prediction_step')

        # backwards compatibility with previously saved models
        name = 'decoder_fw_step_state_c' if bidirectional \
            else 'decoder_step_state_c'
        self.decoder_fw_step_c = tf.placeholder(tf.float32,
                                                [None, lstm_units], name)
        name = 'decoder_fw_step_state_h' if bidirectional \
            else 'decoder_step_state_h'
        self.decoder_fw_step_h = tf.placeholder(tf.float32,
                                                [None, lstm_units], name)
        self.decoder_bw_step_c = tf.placeholder(tf.float32,
                                                [None, lstm_units],
                                                'decoder_bw_step_state_c')
        self.decoder_bw_step_h = tf.placeholder(tf.float32,
                                                [None, lstm_units],
                                                'decoder_bw_step_state_h')

        with tf.variable_scope('autoencoder') as self.scope:
            self.embeddings = tf.Variable(embeddings, name='embeddings',
                                          trainable=train_embeddings)

            initializer = tf.contrib.layers.xavier_initializer()
            self.lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_units,
                                                   initializer=initializer)
            self.lstm_bw = tf.nn.rnn_cell.LSTMCell(lstm_units,
                                                   initializer=initializer)
            shape = (2 * lstm_units, self.vocab_size) if bidirectional \
                else (lstm_units, self.vocab_size)
            self.projection_w = tf.get_variable('projection_w', shape,
                                                initializer=initializer)
            initializer = tf.zeros_initializer((self.vocab_size,))
            self.projection_b = tf.get_variable('projection_b',
                                                initializer=initializer)
            embedded = tf.nn.embedding_lookup(self.embeddings, self.sentence)
            embedded = tf.nn.dropout(embedded, self.dropout_keep)

            if bidirectional:
                bdr = tf.nn.bidirectional_dynamic_rnn
                ret = bdr(self.lstm_fw, self.lstm_bw,
                          embedded, dtype=tf.float32,
                          sequence_length=self.sentence_size,
                          scope=self.scope)
            else:
                ret = tf.nn.dynamic_rnn(self.lstm_fw, embedded,
                                        dtype=tf.float32,
                                        sequence_length=self.sentence_size,
                                        scope=self.scope)
            _, self.encoded_state = ret
            if bidirectional:
                encoded_state_fw, encoded_state_bw = self.encoded_state

                # set the scope name used inside the decoder.
                # maybe there's a more elegant way to do it?
                fw_scope_name = self.scope.name + '/FW'
                bw_scope_name = self.scope.name + '/BW'
            else:
                encoded_state_fw = self.encoded_state
                fw_scope_name = self.scope

            self.scope.reuse_variables()

        if train:
            # seq2seq functions need lists as input
            list_input = self._tensor_to_list(embedded)

            # generate a batch of embedded EOS
            # sentence_size has the batch dimension
            eos_batch = self._generate_batch_eos(self.sentence_size)
            embedded_eos = tf.nn.embedding_lookup(self.embeddings,
                                                  eos_batch)
            decoder_input = [embedded_eos] + list_input

            # We give the same inputs to the forward and backward LSTMs,
            # but each one has its own hidden state
            # their outputs are concatenated and fed to the softmax layer

            # The BW LSTM sees the input in reverse order but make predictions
            # in forward order
            with tf.variable_scope(fw_scope_name, reuse=True) as fw_scope:
                res = tf.nn.seq2seq.rnn_decoder(decoder_input,
                                                encoded_state_fw,
                                                self.lstm_fw,
                                                scope=fw_scope)
                decoder_outputs_fw, _ = res

            if bidirectional:
                with tf.variable_scope(bw_scope_name, reuse=True) as bw_scope:
                    res = tf.nn.seq2seq.rnn_decoder(decoder_input,
                                                    encoded_state_bw,
                                                    self.lstm_bw,
                                                    scope=bw_scope)
                    decoder_outputs_bw, _ = res

            # decoder_outputs has the raw outputs before projection
            # it has shape (batch, lstm_units)
            self.decoder_outputs = []
            raw_outputs = zip(decoder_outputs_fw, decoder_outputs_bw) \
                if bidirectional else decoder_outputs_fw
            for output in raw_outputs:
                if bidirectional:
                    # here, each output is (output_fw, output_bw)
                    output = tf.concat(1, output)
                dropout = tf.nn.dropout(output, self.dropout_keep)
                self.decoder_outputs.append(dropout)

        # tensors for running a model
        embedded_step = tf.nn.embedding_lookup(self.embeddings,
                                               self.decoder_step_input)
        state_fw = tf.nn.rnn_cell.LSTMStateTuple(self.decoder_fw_step_c,
                                                 self.decoder_fw_step_h)
        state_bw = tf.nn.rnn_cell.LSTMStateTuple(self.decoder_bw_step_c,
                                                 self.decoder_bw_step_h)
        with tf.variable_scope(fw_scope_name, reuse=True):
            ret_fw = self.lstm_fw(embedded_step, state_fw)
        step_output_fw, self.decoder_fw_step_state = ret_fw

        if bidirectional:
            with tf.variable_scope(bw_scope_name, reuse=True):
                ret_bw = self.lstm_bw(embedded_step, state_bw)
                step_output_bw, self.decoder_bw_step_state = ret_bw
                step_output = tf.concat(1, [step_output_fw, step_output_bw])
        else:
            step_output = step_output_fw
        self.projected_step_output = tf.nn.xw_plus_b(step_output,
                                                     self.projection_w,
                                                     self.projection_b)

        if train:
            self._create_training_tensors()

    def _tensor_to_list(self, tensor, num_steps=None):
        """
        Splits the input tensor sentence into a list of 1-d
        tensors, as much as the number of time steps.
        This is necessary for seq2seq functions.
        """
        if num_steps is None:
            num_steps = self.num_time_steps
        return [tf.squeeze(step, [1])
                for step in tf.split(1, num_steps, tensor)]

    def _create_training_tensors(self):
        """
        Create member variables related to training.
        """
        sentence_as_list = self._tensor_to_list(self.sentence)
        eos_batch = self._generate_batch_eos(sentence_as_list[0])
        decoder_labels = sentence_as_list + [eos_batch]
        decoder_labels = [tf.cast(step, tf.int64) for step in decoder_labels]

        # set the importance of each time step
        # 1 if before sentence end or EOS itself; 0 otherwise
        label_weights = [tf.cast(tf.less(i - 1, self.sentence_size),
                                 tf.float32)
                         for i in range(self.num_time_steps + 1)]

        projection_w_t = tf.transpose(self.projection_w)

        def loss_function(inputs, labels):
            labels = tf.reshape(labels, (-1, 1))
            return tf.nn.sampled_softmax_loss(projection_w_t,
                                              self.projection_b,
                                              inputs, labels,
                                              100, self.vocab_size)
        labeled_loss = tf.nn.seq2seq.sequence_loss(self.decoder_outputs,
                                                   decoder_labels,
                                                   label_weights,
                                                   softmax_loss_function=loss_function)
        # self.loss = labeled_loss + self.compute_l2_loss()
        self.loss = labeled_loss
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)

        self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                  global_step=self.global_step)

    def get_trainable_variables(self):
        """
        Return all trainable variables inside the model
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 self.scope.name)

    def train(self, session, save_path, train_data, valid_data,
              batch_size, epochs, learning_rate, dropout_keep,
              clip_value, report_interval):
        """
        Train the model

        :param session: tensorflow session
        :param train_data: Dataset object with training data
        :param valid_data: Dataset object with validation data
        :param batch_size: batch size
        :param learning_rate: initial learning rate
        :param dropout_keep: the probability that each LSTM input/output is kept
        :param epochs: how many epochs to train for
        :param clip_value: value to clip tensor norm during training
        :param save_path: folder to save the model
        :param report_interval: report after that many batches
        """
        saver = tf.train.Saver(self.get_trainable_variables(),
                               max_to_keep=1)

        best_loss = 10000
        accumulated_loss = 0
        batch_counter = 0

        # get all data at once. we need all matrices with the same size,
        # or else they don't fit the placeholders
        train_sents, train_sizes = train_data.join_all(self.eos,
                                                       self.num_time_steps,
                                                       shuffle=True)

        del train_data  # save memory...
        valid_sents, valid_sizes = valid_data.join_all(self.eos,
                                                       self.num_time_steps,
                                                       shuffle=True)

        for i in range(epochs):
            # idx1 and idx2 are used to index where each batch begins and
            # ends in train_data
            idx1 = 0
            while idx1 < len(train_sents):
                idx2 = idx1 + batch_size
                feeds = {self.sentence: train_sents[idx1:idx2],
                         self.sentence_size: train_sizes[idx1:idx2],
                         self.clip_value: clip_value,
                         self.dropout_keep: dropout_keep,
                         self.learning_rate: learning_rate}

                _, loss = session.run([self.train_op, self.loss], feeds)
                # tl = timeline.Timeline(run_metadata.step_stats)
                # ctf = tl.generate_chrome_trace_format()
                # with open('timeline-%d.json' % batch_counter, 'wb') as f:
                #     f.write(ctf)

                accumulated_loss += loss

                idx1 = idx2
                batch_counter += 1
                if batch_counter % report_interval == 0:
                    avg_loss = accumulated_loss / report_interval
                    accumulated_loss = 0

                    # we can't use all the validation at once, since it would
                    # take too much memory. running many small batches would
                    # instead take too much time. So let's just sample it.
                    sample_indices = np.random.randint(0, len(valid_data),
                                                       5000)
                    feeds = {self.sentence: valid_sents[sample_indices],
                             self.sentence_size: valid_sizes[sample_indices],
                             self.dropout_keep: 1}

                    loss = session.run(self.loss, feeds)
                    msg = '%d epochs, %d batches\t' % (i, batch_counter)
                    msg += 'Avg batch loss: %f\t' % avg_loss
                    msg += 'Validation loss: %f' % loss
                    if loss < best_loss:
                        best_loss = loss
                        self.save(saver, session, save_path)
                        msg += '\t(saved model)'

                    logging.info(msg)

    def save(self, saver, session, directory):
        """
        Save the autoencoder model and metadata to the specified
        directory.
        """
        model_path = os.path.join(directory, 'model')
        saver.save(session, model_path)
        metadata = {'num_time_steps': self.num_time_steps,
                    'vocab_size': self.vocab_size,
                    'embedding_size': self.embedding_size,
                    'num_units': self.lstm_fw.output_size,
                    'eos': self.eos,
                    'bidirectional': self.bidirectional
                    }
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'wb') as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, directory, session, train=False):
        """
        Load an instance of this class from a previously saved one.
        :param directory: directory with the model files
        :param session: tensorflow session
        :param train: if True, also create training tensors
        :return: a TextAutoencoder instance
        """
        model_path = os.path.join(directory, 'model')
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)
        num_time_steps = metadata['num_time_steps'] if train else None
        dummy_embeddings = np.empty((metadata['vocab_size'],
                                     metadata['embedding_size'],),
                                    dtype=np.float32)

        ae = TextAutoencoder(metadata['num_units'], num_time_steps,
                             dummy_embeddings,
                             metadata['eos'], train=train,
                             bidirectional=metadata['bidirectional'])
        vars_to_load = ae.get_trainable_variables()
        if not train:
            # if not flagged for training, the embeddings won't be in
            # the list
            vars_to_load.append(ae.embeddings)

        saver = tf.train.Saver(vars_to_load)
        saver.restore(session, model_path)
        return ae

    def encode(self, session, inputs, sizes):
        """
        Run the encoder to obtain the encoded hidden state

        :param session: tensorflow session
        :param inputs: 2-d array with the word indices
        :param sizes: 1-d array with size of each sentence
        :return: a 2-d numpy array with the hidden state
        """
        feeds = {self.sentence: inputs,
                 self.sentence_size: sizes,
                 self.dropout_keep: 1}
        state = session.run(self.encoded_state, feeds)
        if self.bidirectional:
            state_fw, state_bw = state
            return np.hstack((state_fw.c, state_bw.c))
        return state.c

    def run(self, session, inputs, sizes):
        """
        Run the autoencoder with the given data

        :param session: tensorflow session
        :param inputs: 2-d array with the word indices
        :param sizes: 1-d array with size of each sentence
        :return: a 2-d array (batch, output_length) with the answer
            produced by the autoencoder. The output length is not
            fixed; it stops after producing EOS for all items in the
            batch or reaching two times the maximum number of time
            steps in the inputs.
        """
        feeds = {self.sentence: inputs,
                 self.sentence_size: sizes,
                 self.dropout_keep: 1}
        state = session.run(self.encoded_state, feeds)
        if self.bidirectional:
            state_fw, state_bw = state
        else:
            state_fw = state

        time_steps = 0
        max_time_steps = 2 * len(inputs[0])
        answer = []
        input_symbol = self.eos * np.ones_like(sizes, dtype=np.int32)

        # this array control which sequences have already been finished by the
        # decoder, i.e., for which ones it already produced the END symbol
        sequences_done = np.zeros_like(sizes, dtype=np.bool)

        while True:
            # we could use tensorflow's rnn_decoder, but this gives us
            # finer control

            feeds = {self.decoder_fw_step_c: state_fw.c,
                     self.decoder_fw_step_h: state_fw.h,
                     self.decoder_step_input: input_symbol,
                     self.dropout_keep: 1}
            if self.bidirectional:
                feeds[self.decoder_bw_step_c] = state_bw.c
                feeds[self.decoder_bw_step_h] = state_bw.h

                ops = [self.projected_step_output,
                       self.decoder_fw_step_state,
                       self.decoder_bw_step_state]
                outputs, state_fw, state_bw = session.run(ops, feeds)
            else:
                ops = [self.projected_step_output,
                       self.decoder_fw_step_state]
                outputs, state_fw = session.run(ops, feeds)

            input_symbol = outputs.argmax(1)
            answer.append(input_symbol)

            # use an "additive" or in order to avoid infinite loops
            sequences_done |= (input_symbol == self.eos)

            if sequences_done.all() or time_steps > max_time_steps:
                break
            else:
                time_steps += 1

        return np.hstack(answer)

    def _generate_batch_eos(self, like):
        """
        Generate a 1-d tensor with copies of EOS as big as the batch size,

        :param like: a tensor whose shape the returned embeddings should match
        :return: a tensor with shape as `like`
        """
        ones = tf.ones_like(like)
        return ones * self.eos
