# -*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf
import numpy as np
import logging
import json
import os


class TextVariationalAutoencoder(object):
    """
    Class that encapsulates the encoder-decoder architecture to
    reconstruct pieces of text.
    """

    def __init__(self, lstm_units, latent_units, embeddings,
                 train_embeddings=False, create_special_embeddings=True):
        """
        Initialize the encoder/decoder and creates Tensor objects

        :param lstm_units: number of LSTM units
        :param latent_units: number of units in the latent representation
        :param embeddings: numpy array with initial embeddings
        :param train_embeddings: whether to adjust embeddings during training
        :param create_special_embeddings: only used internally, and should be
            True when first creating the model. It creates embeddings for
            special tokens as GO and EOS.
        """
        orig_vocab_size = embeddings.shape[0]
        self.embedding_size = embeddings.shape[1]
        self.lstm_units = lstm_units
        self.latent_units = latent_units

        if create_special_embeddings:
            self.go = orig_vocab_size
            self.eos = orig_vocab_size + 1
            new_embeddings = np.random.normal(
                embeddings.mean(0), embeddings.std(0), [2, self.embedding_size])
            embeddings = np.concatenate([embeddings, new_embeddings])
            self.vocab_size = orig_vocab_size + 2
        else:
            self.go = orig_vocab_size - 2
            self.eos = orig_vocab_size - 1
            self.vocab_size = orig_vocab_size

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # the sentence is the object to be memorized
        self.sentence = tf.placeholder(tf.int32, [None, None], 'sentence')
        self.sentence_size = tf.placeholder(tf.int32, [None],
                                            'sentence_size')
        self.l2_constant = tf.placeholder(tf.float32, name='l2_constant')
        self.clip_value = tf.placeholder(tf.float32, name='clip')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')
        self.is_deterministic = tf.placeholder_with_default(0., None,
                                                            'is_deterministic')
        self.kl_coefficient = tf.placeholder_with_default(1., None,
                                                          'kl_loss_coefficient')

        with tf.variable_scope('encoder') as self.encoder_scope:
            self.embeddings = tf.Variable(embeddings, name='embeddings',
                                          trainable=train_embeddings,
                                          dtype=tf.float32)

            initializer = tf.glorot_normal_initializer()
            encoder_lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_units,
                                                      initializer=initializer)
            encoder_lstm_bw = tf.nn.rnn_cell.LSTMCell(lstm_units,
                                                      initializer=initializer)

            embedded = tf.nn.embedding_lookup(self.embeddings, self.sentence)
            embedded = tf.nn.dropout(embedded, self.dropout_keep)

            bdr = tf.nn.bidirectional_dynamic_rnn
            outputs, _ = bdr(encoder_lstm_fw, encoder_lstm_bw, embedded,
                             dtype=tf.float32, scope=self.encoder_scope,
                             sequence_length=self.sentence_size)

            output_fw, output_bw = outputs
            last_output_fw = output_fw[:, -1, :]
            last_output_bw = output_bw[:, -1, :]
            last_output = tf.concat([last_output_fw, last_output_bw], -1)

            self.mean = tf.layers.dense(last_output, latent_units,
                                        kernel_initializer=initializer)
            self.log_sigma = tf.layers.dense(last_output, latent_units,
                                             kernel_initializer=initializer)
            eps = tf.random_normal(tf.shape(self.mean))
            variance = tf.sqrt(tf.exp(self.log_sigma)) * eps
            self.latent_state = self.mean + \
                (1 - self.is_deterministic) * variance

        with tf.variable_scope('decoder') as self.decoder_scope:
            # reshape trick necessary; or else tensorflow can't determine rank
            latent_state = tf.reshape(self.latent_state, tf.shape(self.mean))

            # use 2 * lstm_units to be the same size as the bilstm encoder
            decoder_projection_layer = tf.layers.Dense(2 * self.lstm_units)
            decoder_state = decoder_projection_layer(latent_state)
            state_tuple = tf.nn.rnn_cell.LSTMStateTuple(
                decoder_state, tf.zeros_like(decoder_state))

            # generate a batch of embedded GO
            # sentence_size has the batch dimension
            go_batch = self._generate_batch_symbol(self.sentence_size, self.go)
            embedded_go = tf.nn.embedding_lookup(self.embeddings,
                                                 go_batch)
            embedded_go = tf.reshape(embedded_go,
                                     [-1, 1, self.embedding_size])
            decoder_input = tf.concat([embedded_go, embedded], axis=1)

            decoder_lstm = tf.nn.rnn_cell.LSTMCell(2 * lstm_units,
                                                   initializer=initializer)

            outputs, _ = tf.nn.dynamic_rnn(
                decoder_lstm, decoder_input, self.sentence_size,
                state_tuple, scope=self.decoder_scope)

            self.decoder_outputs = outputs
            softmax_layer = tf.layers.Dense(self.vocab_size, name='softmax')
            self.logits = softmax_layer(outputs)

        ###############################
        # Tensors for running a model #
        ###############################
        self.decoder_step_input = tf.placeholder(tf.int32,
                                                 [None],
                                                 'prediction_step')
        self.decoder_step_h = tf.placeholder(tf.float32, [None, 2 * lstm_units],
                                             'decoder_step_state_h')
        self.decoder_step_c = tf.placeholder(tf.float32, [None, 2 * lstm_units],
                                             'decoder_step_state_c')
        self.latent_state_input = tf.placeholder(
            tf.float32, [None, latent_units], 'latent_state_step')

        embedded_step = tf.nn.embedding_lookup(self.embeddings,
                                               self.decoder_step_input)
        self.decoder_state = decoder_state
        self.decoder_state_from_latent = decoder_projection_layer(
            self.latent_state_input)

        decoder_step_state = tf.nn.rnn_cell.LSTMStateTuple(
            self.decoder_step_c, self.decoder_step_h)
        ret = decoder_lstm(embedded_step, decoder_step_state,
                           scope=self.decoder_scope)
        step_output, (self.next_decoder_c, self.next_decoder_h) = ret
        self.step_logits = softmax_layer(step_output)

        self._create_training_tensors()

    def _create_training_tensors(self):
        """
        Create member variables related to training.
        """
        eos_batch = self._generate_batch_symbol(self.sentence_size, self.eos)
        eos_batch = tf.reshape(eos_batch, [-1, 1])
        decoder_labels = tf.concat([self.sentence, eos_batch], -1)

        projection_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=self.decoder_scope.name)
        # a bit ugly, maybe we should improve this?
        output_w = [var for var in projection_vars
                    if 'softmax/kernel' in var.name][0]
        output_b = [var for var in projection_vars
                    if 'softmax/bias' in var.name][0]

        # set the importance of each time step
        # 1 if before sentence end or EOS itself; 0 otherwise
        max_len = tf.shape(self.sentence)[1]
        mask = tf.sequence_mask(self.sentence_size + 1, max_len + 1, tf.float32)
        num_actual_labels = tf.reduce_sum(mask)
        output_w_t = tf.transpose(output_w)

        # reshape to have batch and time steps in the same dimension
        decoder_outputs2d = tf.reshape(self.decoder_outputs,
                                       [-1, tf.shape(self.decoder_outputs)[-1]])
        labels = tf.reshape(decoder_labels, [-1, 1])
        sampled_loss = tf.nn.sampled_softmax_loss(
            output_w_t, output_b, labels, decoder_outputs2d, 100,
            self.vocab_size)

        masked_loss = tf.reshape(mask, [-1]) * sampled_loss

        # KL loss, i.e., the divergence between the latent representation and
        # a normal distribution with mean 0 and sigma 1
        kl_loss = -0.5 * tf.reduce_sum(1 + self.log_sigma -
                                       self.mean ** 2 - tf.exp(self.log_sigma))

        # labeled loss, i.e., reconstructing the input
        reconstruction_loss = tf.reduce_sum(masked_loss) / num_actual_labels

        self.loss = self.kl_coefficient * kl_loss + reconstruction_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)

        self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                  global_step=self.global_step)

    def get_trainable_variables(self):
        """
        Return all trainable variables inside the model
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def train(self, session, save_path, train_data, valid_data,
              batch_size, epochs, learning_rate, dropout_keep,
              clip_value, report_interval, increase_kl_every=10):
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
        :param increase_kl_every: how many batches to wait before increasing
            the kl weight in the loss by 0.01 (it starts at 0 and caps at 1)
        """
        saver = tf.train.Saver(self.get_trainable_variables(),
                               max_to_keep=1)

        best_loss = 10000
        accumulated_loss = 0
        batch_counter = 0
        num_sents = 0
        kl_coefficient = 0

        # get all data at once. we need all matrices with the same size,
        # or else they don't fit the placeholders
        # train_sents, train_sizes = train_data.join_all(self.go,
        #                                                self.num_time_steps,
        #                                                shuffle=True)

        # del train_data  # save memory...
        valid_sents, valid_sizes = valid_data.join_all(self.eos,
                                                       shuffle=True)
        train_data.reset_epoch_counter()
        feeds = {self.clip_value: clip_value,
                 self.dropout_keep: dropout_keep,
                 self.learning_rate: learning_rate,
                 self.kl_coefficient: kl_coefficient}

        while train_data.epoch_counter < epochs:
            batch_counter += 1
            train_sents, train_sizes = train_data.next_batch(batch_size)
            feeds[self.sentence] = train_sents
            feeds[self.sentence_size] = train_sizes

            _, loss = session.run([self.train_op, self.loss], feeds)

            # multiply by len because some batches may be smaller
            # (due to bucketing), then take the average
            accumulated_loss += loss * len(train_sents)
            num_sents += len(train_sents)

            if batch_counter % report_interval == 0:
                avg_loss = accumulated_loss / num_sents
                accumulated_loss = 0
                num_sents = 0

                # we can't use all the validation at once, since it would
                # take too much memory. running many small batches would
                # instead take too much time. So let's just sample it.
                sample_indices = np.random.randint(0, len(valid_data),
                                                   5000)
                validation_feeds = {
                    self.sentence: valid_sents[sample_indices],
                    self.sentence_size: valid_sizes[sample_indices],
                    self.dropout_keep: 1}

                loss = session.run(self.loss, validation_feeds)
                msg = '%d epochs, %d batches\t' % (train_data.epoch_counter,
                                                   batch_counter)
                msg += 'Avg batch loss: %f\t' % avg_loss
                msg += 'Validation loss: %f\t' % loss
                msg += 'KL weight: %.2f' % kl_coefficient
                if loss < best_loss:
                    best_loss = loss
                    self.save(saver, session, save_path)
                    msg += '\t(saved model)'

                logging.info(msg)

            if batch_counter % increase_kl_every == 0 and kl_coefficient < 1:
                kl_coefficient += 0.01
                feeds[self.kl_coefficient] = kl_coefficient

    def save(self, saver, session, directory):
        """
        Save the autoencoder model and metadata to the specified
        directory.
        """
        model_path = os.path.join(directory, 'model')
        saver.save(session, model_path)
        metadata = {'vocab_size': self.vocab_size,
                    'embedding_size': self.embedding_size,
                    'lstm_units': self.lstm_units,
                    'latent_units': self.latent_units
                    }
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, directory, session):
        """
        Load an instance of this class from a previously saved one.
        :param directory: directory with the model files
        :param session: tensorflow session
        :return: a TextAutoencoder instance
        """
        model_path = os.path.join(directory, 'model')
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        dummy_embeddings = np.empty((metadata['vocab_size'],
                                     metadata['embedding_size'],),
                                    dtype=np.float32)

        # embeddings must be trained to be loaded...
        # TODO: change to load them regardless
        ae = TextVariationalAutoencoder(
            metadata['lstm_units'], metadata['latent_units'], dummy_embeddings,
            train_embeddings=True, create_special_embeddings=False)
        vars_to_load = ae.get_trainable_variables()

        saver = tf.train.Saver(vars_to_load)
        saver.restore(session, model_path)
        return ae

    def encode(self, session, inputs, sizes, deterministic=True):
        """
        Run the encoder to obtain the encoded latent state

        :param session: tensorflow session
        :param inputs: 2-d array with the word indices
        :param sizes: 1-d array with size of each sentence
        :param deterministic: whether the output should be deterministic (using
            only the mean of the distribution, or include the variance).
        :return: a 2-d numpy array with the latent state, shape
            (num_sents, latent_size)
        """
        feeds = {self.sentence: inputs,
                 self.sentence_size: sizes,
                 self.dropout_keep: 1}
        op = self.latent_state if deterministic else self.mean
        state = session.run(op, feeds)

        return state

    def run(self, session, inputs, sizes, deterministic=True):
        """
        Run the autoencoder with the given data

        :param session: tensorflow session
        :param inputs: 2-d array with the word indices
        :param sizes: 1-d array with size of each sentence
        :param deterministic: whether the output should be deterministic (using
            only the mean of the distribution, or include the variance).
        :return: a 2-d array (batch, output_length) with the answer
            produced by the autoencoder. The output length is not
            fixed; it stops after producing EOS for all items in the
            batch or reaching two times the maximum number of time
            steps in the inputs.
        """
        is_deterministic_value = 1. if deterministic else 0.
        feeds = {self.sentence: inputs,
                 self.sentence_size: sizes,
                 self.dropout_keep: 1,
                 self.is_deterministic: is_deterministic_value}

        # state is (num_sents, lstm_size)
        state_c = session.run(self.decoder_state, feeds)
        state_h = np.zeros_like(state_c)

        time_steps = 0
        max_time_steps = 2 * len(inputs[0])
        answer = []
        input_symbol = self.go * np.ones_like(sizes, dtype=np.int32)

        # this array control which sequences have already been finished by the
        # decoder, i.e., for which ones it already produced the END symbol
        sequences_done = np.zeros_like(sizes, dtype=np.bool)

        while True:
            feeds = {self.decoder_step_c: state_c,
                     self.decoder_step_h: state_h,
                     self.decoder_step_input: input_symbol,
                     self.dropout_keep: 1,
                     self.is_deterministic: is_deterministic_value}

            ops = [self.step_logits,
                   self.next_decoder_c,
                   self.next_decoder_h]
            logits, state_c, state_h = session.run(ops, feeds)

            input_symbol = logits.argmax(1)
            answer.append(input_symbol)

            # use an "additive" OR operator in order to avoid infinite loops
            sequences_done |= (input_symbol == self.eos)

            if sequences_done.all() or time_steps > max_time_steps:
                break
            else:
                time_steps += 1

        return np.hstack(answer)

    def _generate_batch_symbol(self, like, symbol):
        """
        Generate a 1-d tensor with copies of a given symbol (such as EOS or go)

        :param like: a tensor whose shape the returned embeddings should match
        :param symbol: the value that every item in the batch must have
        :return: a tensor with shape as `like`
        """
        ones = tf.ones_like(like)
        return ones * symbol
