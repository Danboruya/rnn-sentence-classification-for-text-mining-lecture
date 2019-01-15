import tensorflow as tf
from sklearn import metrics
import numpy as np
import os
import time
import csv
import datetime
import data_controller
from model import Model

# ==================
# Parameter settings
# ==================

flags = tf.flags

# ==Data==
flags.DEFINE_string("positive_train_data_path", "./data/tw-polaritydata/tw-train.pos",
                    "File path for the positive train data.")
flags.DEFINE_string("negative_train_data_path", "./data/tw-polaritydata/tw-train.neg",
                    "File path for the negative train data.")
flags.DEFINE_string("positive_test_data_path", "./data/tw-polaritydata/tw-test.pos",
                    "File path for the positive test data.")
flags.DEFINE_string("negative_test_data_path", "./data/tw-polaritydata/tw-test.neg",
                    "File path for the negative test data.")

# ==Hyper parameters==
flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of word embedding")
flags.DEFINE_integer("n_cell", 32, "The number of unit of cell")
flags.DEFINE_integer("n_layer", 2, "The number of hidden layer")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
flags.DEFINE_float("learning_rate", 1e-6, "Learning rate")
flags.DEFINE_integer("n_class", 2, "The number of classifier")
flags.DEFINE_float("f_bias", 1.0, "Forget bias")
flags.DEFINE_string("cell_type", "GRU", "The type of cell on hidden layer")
flags.DEFINE_string("filter_sizes", "3,4,4", "The size of cnn filter")
flags.DEFINE_integer("n_filter", 4, "The number of filter par filter size")

# ==Training parameters==
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("n_epoch", 300, "The number of training epochs")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps")
flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store")

# ==Other parameters==
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
flags.DEFINE_string("exp_name", "exp1", "Experiment name")

FLAGS = flags.FLAGS


def load_train_data():
    """
    Loading the train data from data set
    :return: Training components
    """
    # Data preparation for train and test
    raw_data_set = data_controller.load_data_file(FLAGS.positive_train_data_path,
                                                  FLAGS.negative_train_data_path,
                                                  FLAGS.positive_test_data_path,
                                                  FLAGS.negative_test_data_path)
    sentence_length = data_controller.get_sentence_length(raw_data_set.all_train_data_set,
                                                          raw_data_set.all_test_data_set)
    vocab_train_data, raw_train_input_data = data_controller.build_vocabulary(raw_data_set.positive_train_data,
                                                                              raw_data_set.negative_train_data,
                                                                              raw_data_set.all_train_data_set,
                                                                              sentence_length)
    vocab_test_data, raw_test_input_data = data_controller.build_vocabulary(raw_data_set.positive_test_data,
                                                                            raw_data_set.negative_test_data,
                                                                            raw_data_set.all_test_data_set,
                                                                            sentence_length)

    # data_set[0]/label[0]:Train, data_set[1]/label[1]:Test
    data_set, data_set_label = data_controller.data_divider(raw_train_input_data[0], raw_data_set.data_set_train_label)
    x_test, y_test, x_raw_test = data_controller.data_shuffler(raw_test_input_data[0], raw_data_set.data_set_test_label,
                                                               raw_data_set.all_test_data_set)
    # x_raw_test = raw_data_set.all_test_data_set

    x_train = data_set[0]
    y_train = data_set_label[0]
    x_valid = data_set[1]
    y_valid = data_set_label[1]
    sentence_length = raw_train_input_data[5]
    n_class = FLAGS.n_class
    vocab_processor = vocab_train_data[3]

    print_info(vocab_processor, y_train, y_valid, y_test)

    components = [x_train, y_train, x_valid, y_valid,
                  sentence_length, n_class, vocab_processor,
                  x_test, y_test, x_raw_test]
    return components


def print_info(vocab_processor, y_train, y_valid, y_test):
    print("===================")
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)))
    print("Test: {:d}".format(len(y_test)))
    print("Parameters:")
    for key in FLAGS.__flags.keys():
        print('{}={}'.format(key, getattr(FLAGS, key)))
    print("===================")


def train(x_train, y_train, x_valid, y_valid, sentence_length, n_class, vocab_processor):
    """
    Training the model
    :param x_train: Input data for training step
    :param y_train: Label data for training step
    :param x_valid: Input data for validation step
    :param y_valid: Label data for validation step
    :param sentence_length: Max sentence length
    :param n_class: The number of classification category
    :param vocab_processor: Object of vocabulary processor
    :return: Output directory path of result of training
    """
    with tf.Graph().as_default():
        session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_config)
        with sess.as_default():
            net = Model(
                sentence_length=sentence_length,
                n_class=n_class,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                n_unit=FLAGS.n_cell,
                n_layer=FLAGS.n_layer,
                cell_type=FLAGS.cell_type,
                f_bias=FLAGS.f_bias,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                n_filter=FLAGS.n_filter
            )

            print("Network instance has been created")

            # Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            print("Optimizer has been set")
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_optimizer = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            print("Complected training procedure settings")

            # Define summaries settings for TensorBoard
            gradient_summaries = []
            for grad, var in grads_and_vars:
                if grad is not None:
                    gradient_histogram_summary = tf.summary.histogram("{}/gradient/histogram".format(var.name), grad)
                    sparsity_summary = tf.summary.scalar("{}/gradient/sparsity".format(var.name),
                                                         tf.nn.zero_fraction(grad))
                    gradient_summaries.append(gradient_histogram_summary)
                    gradient_summaries.append(sparsity_summary)
            gradient_summaries_merged = tf.summary.merge(gradient_summaries)

            # Output directory for model and summaries
            timestamp = str(int(time.time()))
            output_directory = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.cell_type +
                                                            "_" + FLAGS.exp_name + "_" + timestamp))
            print("Writing to {}".format(output_directory))

            # Loss and accuracy summary
            loss_summary = tf.summary.scalar("loss", net.loss)
            accuracy_summary = tf.summary.scalar("accuracy", net.accuracy)

            # Train mini batch summaries
            train_mini_batch_summary_op = tf.summary.merge([loss_summary, accuracy_summary, gradient_summaries_merged])
            train_mini_batch_summary_dir = os.path.join(output_directory, "summaries", "train_mini_bach")
            train_mini_batch_summary_writer = tf.summary.FileWriter(train_mini_batch_summary_dir, sess.graph)
            print("Train mini batch summary has been set")

            # Validate summaries
            validate_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
            validate_summary_dir = os.path.join(output_directory, "summaries", "validate")
            validate_summary_writer = tf.summary.FileWriter(validate_summary_dir, sess.graph)
            print("Validation summary has been set")

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(output_directory, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            print("Checkpoint file write to {}".format(checkpoint_dir))
            print("Checkpoint directory has been set")

            # Save vocabulary
            vocab_processor.save(os.path.join(output_directory, "vocab"))
            print("Vocabulary has been saved")

            # Initialize all variables for TensorFlow
            sess.run(tf.global_variables_initializer())
            print("Boot Session")

            def train_step(x_batch, y_batch, current_epoch):
                """
                A single training step
                :param x_batch: Batch for input data
                :param y_batch: Batch for output data
                :param current_epoch: Current epoch
                """
                feed_dict = {
                    net.input_x: x_batch,
                    net.input_y: y_batch,
                    net.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy, output = sess.run(
                    [train_optimizer, global_step, train_mini_batch_summary_op, net.loss,
                     net.accuracy, net.output],
                    feed_dict)
                train_mini_batch_summary_writer.add_summary(summaries, step)
                print("step {}, epoch {}, loss {:g}, accuracy {:g}".format(step, current_epoch, loss, accuracy))

            def valid_step(_x_valid, _y_valid, writer=None):
                """
                Validate model on a validation set
                :param _x_valid: input data
                :param _y_valid: label data
                :param writer: Summary writer
                """
                feed_dict = {
                    net.input_x: _x_valid,
                    net.input_y: _y_valid,
                    net.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, validate_summary_op, net.loss, net.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Validation")
                print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Training loop
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_valid = np.array(x_valid)
            y_valid = np.array(y_valid)
            n_data = len(x_train)
            current_step = 0
            for epoch in range(FLAGS.n_epoch):
                sff_idx = np.random.permutation(n_data)
                for idx in range(0, n_data, FLAGS.batch_size):
                    batch_x = x_train[
                        sff_idx[idx: idx + FLAGS.batch_size if idx + FLAGS.batch_size < n_data else n_data]]
                    batch_t = y_train[
                        sff_idx[idx: idx + FLAGS.batch_size if idx + FLAGS.batch_size < n_data else n_data]]
                    if len(batch_t) is FLAGS.batch_size:
                        train_step(batch_x, batch_t, epoch)
                        current_step = tf.train.global_step(sess, global_step)
                if epoch == 0 or epoch % 5 == 0 or epoch == (FLAGS.n_epoch - 1):
                    print("============")
                    # train_step(x_train, y_train, epoch, is_batch=False)
                    valid_step(x_valid, y_valid, writer=validate_summary_writer)
                    print("============")
                # if current_step % FLAGS.checkpoint_every == 0:
                if epoch % FLAGS.checkpoint_every == 0 or epoch == (FLAGS.n_epoch - 1):
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}".format(path))
                # saver.save(sess, os.path.join(output_directory, "model.ckpt"), global_step=current_step)
            output_directories = [output_directory, checkpoint_dir]
            return output_directories


def test(x_test, y_test, x_raw_test, out_dir, check_dir, vocab_processor, y_train, y_valid):
    """
    Test the model using non-trained test data.
    :param x_test: Input data for testing steps
    :param y_test: Label data for testing steps
    :param x_raw_test: Raw sentence like human readable for test data
    :param out_dir: Output directory path of saved model
    :param check_dir: Checkpoint directory path of saved model
    :param vocab_processor: Vocabulary object in training data
    :param y_train: Input data of training steps for showing the information
    :param y_valid: Label data of training steps for showing the information
    """
    output_dir_path = out_dir
    checkpoint_file = tf.train.latest_checkpoint(check_dir)
    print("Checkpoint path: " + check_dir)
    graph = tf.Graph()
    print("Evaluate model")
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Restore the model
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("Output_layer/predictions").outputs[0]
            scores = graph.get_operation_by_name("Output_layer/scores").outputs[0]
            all_predictions = []
            all_probabilities = None

            # Test process
            test_data = data_controller.test_data_provider(x_test, FLAGS.batch_size)
            for sentence in test_data:
                sentence_predictions, sentence_scores = sess.run([predictions, scores],
                                                                 {input_x: sentence, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, sentence_predictions])
                probabilities = data_controller.softmax(sentence_scores)
                if all_probabilities is not None:
                    all_probabilities = np.concatenate([all_probabilities, probabilities])
                else:
                    all_probabilities = probabilities

    if y_test is not None:
        print("Total number of test data: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(float(sum(all_predictions == y_test)) / float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw_test), all_predictions))
    csv_out_path = os.path.join(output_dir_path, "prediction.csv")
    print("Saving evaluation to {0}".format(csv_out_path))
    with open(csv_out_path, "w+") as f:
        csv.writer(f).writerows(predictions_human_readable)
    spec_out_path = os.path.join(output_dir_path, "spec.txt")
    result_out_path = os.path.join(output_dir_path, "eval_out.txt")
    test_result_out_path = os.path.join(output_dir_path, "result_out.txt")

    # File out
    with open(spec_out_path, "w+") as f:
        f.write("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)) + "\n")
        f.write("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)) + "\n")
        f.write("Total number of test data: {}".format(len(y_test)) + "\n")
        f.write("============================" + "\n")
        f.write("Parameters:" + "\n")
        for key in FLAGS.__flags.keys():
            f.write('{}={}'.format(key, getattr(FLAGS, key)) + "\n")

    with open(result_out_path, "w+") as f:
        f.write("============================" + "\n")
        f.write("Classification Result" + "\n")
        f.write(str(metrics.classification_report(y_test, all_predictions,
                                                  target_names=["Positive sentence", "Negative sentence"])) + "\n")
        f.write("============================" + "\n")
        f.write("confusion matrix:" + "\n")
        f.write(str(metrics.confusion_matrix(y_test, all_predictions)) + "\n")
        f.write("============================" + "\n")
        if y_test is not None:
            f.write("Total number of test data: {}".format(len(y_test)) + "\n")
            f.write("Accuracy: {:g}".format(float(sum(all_predictions == y_test)) / float(len(y_test))) + "\n")

    with open(test_result_out_path, "w+") as f:
        for idx in range(len(predictions_human_readable)):
            if y_test[idx] == 0:
                label = "{}:Positive".format(y_test[idx])
            else:
                label = "{}:Negative".format(y_test[idx])
            f.write("Classification : " + str(predictions_human_readable[idx]) + ", Correct Label: " + label + "\n")


def main(argv):
    """
    Main function for this program.
    Load train data -> train -> Load test data -> test
    """
    components = load_train_data()
    x_train = components[0]
    y_train = components[1]
    x_valid = components[2]
    y_valid = components[3]
    x_test = components[7]
    x_test_raw = components[9]
    y_test = components[8]
    sentence_length = components[4]
    n_class = components[5]
    vocab_processor = components[6]
    out_dir = train(x_train, y_train, x_valid, y_valid, sentence_length, n_class, vocab_processor)
    test(x_test, y_test, x_test_raw, out_dir[0], out_dir[1], vocab_processor, y_train, y_valid)
    print_info(vocab_processor, y_train, y_valid, y_test)
    print("App has been done")


if __name__ == "__main__":
    tf.app.run()
