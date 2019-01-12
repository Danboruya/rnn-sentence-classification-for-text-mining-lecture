import sys
import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    """
    Net work model for Sentence classification.
    """
    def __init__(self, sentence_length, n_class, vocab_size,
                 embedding_size, n_unit, n_layer, cell_type,
                 f_bias, filter_sizes, n_filter):
        """
        Initialize the instance of this class.
        :param sentence_length: Max length of sentence
        :param n_class: The number of class for classier
        :param vocab_size: The number of vocabulary
        :param embedding_size: Word embedding size
        :param n_unit: The number of unit on LSTM cell
        :param n_layer: The number of hidden layer
        :param cell_type: The type of cell on hidden layer
        :param f_bias: Forget bias
        :param filter_sizes: CNN filter size
        :param n_filter: The number of filter
        """

        # Set Placeholders for input layer, output layer and dropout
        self.input_x = tf.placeholder(tf.int32, shape=(None, sentence_length), name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, n_class], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=None, name="keep_prob")
        l2_loss = tf.constant(0.0)
        l2_reg_lambda = 0.0

        # Word embedding layer as input layer
        with tf.name_scope("Input_layer"):
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="embeddings")
            self.embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedded_chars_exp = tf.expand_dims(self.embedded_chars, -1)
        print("Embedding: Done")

        # Create RNN as hidden layer
        with tf.name_scope("Hidden_layer"):
            if cell_type != "CNN":
                # Create cell
                stacked_rnn = []
                for _ in range(n_layer):
                    if cell_type == "RNN":
                        cell = rnn.BasicRNNCell(n_unit)
                    elif cell_type == "LSTM":
                        cell = rnn.BasicLSTMCell(n_unit, forget_bias=f_bias)
                    elif cell_type == "PLSTM":
                        cell = rnn.LSTMCell(n_unit, forget_bias=f_bias)
                    elif cell_type == "GRU":
                        cell = rnn.GRUCell(n_unit)
                    else:
                        print("CellTypeError: Can not set cell on hidden layer: " + str(cell_type))
                        sys.exit()
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.dropout_keep_prob)
                    stacked_rnn.append(cell)
                cell = rnn.MultiRNNCell(cells=stacked_rnn)
                self.state = cell.zero_state(tf.shape(self.embedded_chars)[0], tf.float32)
                print(self.state)
                outputs = []
                with tf.variable_scope(cell_type):
                    for time_step in range(sentence_length):
                        if time_step > 0:
                            tf.get_variable_scope().reuse_variables()
                        cell_output, self.state = cell(self.embedded_chars[:, time_step, :], self.state)
                        outputs.append(cell_output)
                self.output = outputs[-1]
                print(self.output)
            else:
                self.pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.name_scope("Convolutional-max-pooling-{}".format(filter_size)):
                        # Convolution Layer
                        with tf.name_scope("Convolution_layer"):
                            filter_shape = [filter_size, embedding_size, 1, n_filter]
                            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                            b = tf.Variable(tf.constant(0.1, shape=[n_filter]), name="b")
                            convolution = tf.nn.conv2d(self.embedded_chars_exp, w, strides=[1, 1, 1, 1],
                                                       padding="VALID", name="convolution")
                            # Apply non linearity
                            h = tf.nn.relu(tf.nn.bias_add(convolution, b), name="relu")

                        # Max-pooling over the outputs
                        with tf.name_scope("Pooling_layer"):
                            pooled = tf.nn.max_pool(h, ksize=[1, sentence_length - filter_size + 1, 1, 1],
                                                    strides=[1, 1, 1, 1], padding='VALID', name="pool")
                        self.pooled_outputs.append(pooled)
                self.n_filter_total = n_filter * len(filter_sizes)
                self.h_pool = tf.concat(self.pooled_outputs, 3)
                self.output = tf.reshape(self.h_pool, [-1, self.n_filter_total])
        print("Hidden layer: Done")

        # Dropout in hidden layer
        if cell_type == "CNN":
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.output, keep_prob=self.dropout_keep_prob)
            print("Dropout: Done")

        # Output layer
        with tf.name_scope("Output_layer"):
            if cell_type != "CNN":
                w = tf.get_variable("w", [n_unit, n_class])
                b = tf.get_variable("b", [n_class])
                # self.scores = tf.nn.xw_plus_b(self.output, w, b, name="scores")
                self.scores = tf.nn.softmax(tf.matmul(self.output, w) + b, name="scores")
                self.predictions = tf.argmax(self.scores, axis=1, name="predictions")
            else:
                w = tf.get_variable("w", shape=[self.n_filter_total, n_class],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[n_class]), name="b")
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
        print("Output: Done")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # self.loss = tf.reduce_mean(losses, name="Cross-entropy")
            # self.loss = - tf.reduce_sum(self.scores * tf.log(self.input_y)
            #                            + (1 - self.scores) * tf.log(1 - self.input_y), name="Cross-entropy")
            if cell_type != "CNN":
                self.loss = tf.reduce_mean(tf.square(self.input_y - self.scores), name="Square-loss")
            else:
                # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores,
                #                                                                           labels=self.input_y) +
                #                            l2_reg_lambda * l2_loss, name="Softmax-cross-entropy")
                self.loss = tf.reduce_mean(tf.square(self.input_y - self.scores), name="Square-loss")
        print("Loss: Done")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        print("Accuracy: Done")
        print("==Network construction has been finished==")
