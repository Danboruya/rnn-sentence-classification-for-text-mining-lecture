import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import csv
import os
import sys
import time
import data_controller
from decimal import Decimal

# ==================
# Parameter settings
# ==================

flags = tf.flags

# ==Data==
flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity-utf8.pos",
                    "Data source for the positive data.")
flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity-utf8.neg",
                    "Data source for the negative data.")

# ==Training parameters==
flags.DEFINE_integer("batch_size", 32, "Batch size")

# ==Other parameters==
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
flags.DEFINE_string("out_dir", "", "Output directory")
flags.DEFINE_string("log_mode", False, "Allow to display the test result")
flags.DEFINE_string("application_version", "alpha 0.0.1", "Application version information")

flags.DEFINE_string("mode", "Interactive", "Switch Interactive shell mode or predefined test statement mode")

FLAGS = flags.FLAGS


def load_test_data(x_raw, y_test, out_dir):
    """
    Load data for test
    :param x_raw: Raw data of test sentence
    :param y_test: Label data of test sentence
    :param out_dir: Output directory
    :return: Test components
    """
    # print("===================")
    # print("Evaluating process")
    vocab_path = os.path.join(out_dir, "vocab")
    # print("Vocab_path: " + vocab_path)
    # print("Output dir: " + out_dir)
    test_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(test_vocab_processor.transform(x_raw)))
    _test_components = [x_test, y_test, x_raw]
    return _test_components


def test(x_test, y_test, x_raw, out_dir, mode):
    """
    Test process
    :param x_test: Input data for testing
    :param y_test: Label data for testing
    :param x_raw: Original sentence of test data
    :param out_dir: Output directory path
    :param mode: Allow test mode
    :return Result of test
    """
    output_dir_path = out_dir
    check_dir = os.path.join(out_dir, "checkpoints")
    checkpoint_file = tf.train.latest_checkpoint(check_dir)
    # print("Checkpoint path: " + check_dir)
    # print("===================")
    graph = tf.Graph()
    # print("Load model")
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Restore the model
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("Output_layer/predictions").outputs[0]
            all_predictions = []

            # Test process
            # print("===================")
            # print("Test model")
            test_data = data_controller.test_data_provider(x_test, FLAGS.batch_size)
            for sentence in test_data:
                sentence_predictions = sess.run(predictions, {input_x: sentence, dropout_keep_prob: 1.0})
                # print(sentence_predictions)
                all_predictions = np.concatenate([all_predictions, sentence_predictions])
                # print(all_predictions)
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test data: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    # timestamp = str(int(time.time()))
    # csv_out_path = os.path.join(output_dir_path, timestamp + "_prediction.csv")
    # if mode:
    #     print("Network predictions are " + str(all_predictions))
    #     for idx in range(len(predictions_human_readable)):
    #         print("Test result : " + str(predictions_human_readable[idx]) + ", Correct Label: " + str(y_test[idx]))
    # print("Saving evaluation to {0}".format(csv_out_path))
    # with open(csv_out_path, "w+") as f:
    #     csv.writer(f).writerows(predictions_human_readable)
    # print(predictions_human_readable)
    return predictions_human_readable


def interactive_interface():
    print("Welcome to interactive interface.")
    print("Version: {}".format(FLAGS.application_version))
    while True:
        print("Please enter the sentence you want to classify")
        print(">> ", end='')
        statement = str(input())
        if statement == "exit":
            print("This application has been ended.")
            sys.exit()
        # eval_component = _load_data(statement, out_dir=FLAGS.out_dir)
        eval_component = load_test_data([statement], None, out_dir=FLAGS.out_dir)
        # result = _eval(eval_component[0], None, eval_component[1], FLAGS.out_dir)
        result = test(eval_component[0], None, eval_component[2], FLAGS.out_dir, None)
        # print("===================")
        # print("Classification result")
        for combination in result:
            if Decimal(0.5) < Decimal(combination[1]) <= Decimal(1.0):
                print("\"" + combination[0] + "\" is " + "positive sentence.")
            elif Decimal(combination[1]) == Decimal(0.5):
                print("\"" + combination[0] + "\" is " + "neutral sentence.")
            elif Decimal(0.0) <= Decimal(combination[1]) < Decimal(0.5):
                print("\"" + combination[0] + "\" is " + "negative sentence.")
            else:
                print("Error: Input data could not classify. Try another sentence.")
        # print("===================")
        print("\n")


def main(_):
    """
    Main function for Classification app
    """
    if FLAGS.mode != "Interactive":
        # Test data. Test label is; 0:Negative, 1:Positive
        test_sentences = ["a masterpiece four years in the making",
                          "everything is off.",
                          "I agree with you.",
                          "That's to bad",
                          "I succeeded in uninstalling."]
        test_correct_labels = [1, 0, 1, 0, 1]
        test_components = load_test_data(test_sentences, test_correct_labels, FLAGS.out_dir)
        print(test_components)
        result = test(test_components[0], test_components[1], test_components[2],
                      FLAGS.out_dir, FLAGS.log_mode)
        print("===================")
        print("Classification result")
        for combination in result:
            if Decimal(0.5) < Decimal(combination[1]) <= Decimal(1.0):
                print("\"" + combination[0] + "\" is " + "positive sentence.")
            elif Decimal(combination[1]) is Decimal(0.5):
                print("\"" + combination[0] + "\" is " + "neutral sentence.")
            elif Decimal(0.0) <= Decimal(combination[1]) < Decimal(0.5):
                print("\"" + combination[0] + "\" is " + "negative sentence.")
            else:
                print("Error: Input data could not classify. Try another sentence.")
        print("===================")
        sys.exit()
    interactive_interface()


if __name__ == '__main__':
    tf.app.run()
