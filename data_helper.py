import csv
import MeCab
import os


def writer(train_pos, train_neg, test_pos, test_neg):
    """
    File writer
    :param train_pos: Positive dataset for training steps
    :param train_neg: Negative dataset for training steps
    :param test_pos: Positive dataset for testing steps
    :param test_neg: Negative dataset for testing steps
    """
    if not os.path.exists('data/tw-polaritydata'):
        os.mkdir('data/tw-polaritydata')

    with open('data/tw-polaritydata/tw-train.pos', mode='w') as f:
        for sentence in train_pos:
            f.write(str(sentence) + "\n")
    with open('data/tw-polaritydata/tw-train.neg', mode='w') as f:
        for sentence in train_neg:
            f.write(str(sentence) + "\n")
    with open('data/tw-polaritydata/tw-test.pos', mode='w') as f:
        for sentence in test_pos:
            f.write(str(sentence) + "\n")
    with open('data/tw-polaritydata/tw-test.neg', mode='w') as f:
        for sentence in test_neg:
            f.write(str(sentence) + "\n")


def parse(sentences):
    """
    Japanese language parser using MeCab
    (Leaving space between words)
    :param sentences: Set of text
    :return: Result og leaving a space between words
    """
    # For macOS environment
    tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati")

    # For Linux environment
    # tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd -Owakati")

    data = []
    for data_ in sentences:
        parsed_sentence = tagger.parse(data_)
        if parsed_sentence == "\n":
            pass
        else:
            data.append(parsed_sentence.rstrip())
    return data


def main():
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open("data/tweets_sentiment.10000.train.tsv", newline="") as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            if row[5] == "1":
                train_pos.append(row[3])
            else:
                train_neg.append(row[3])

    with open("data/tweets_sentiment.10000.test.tsv", newline="") as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            if row[5] == "1":
                test_pos.append(row[3])
            else:
                test_neg.append(row[3])

    # Check each data size
    print("Positive->Train: {}".format(len(train_pos)))
    print("Negative->Train: {}".format(len(train_neg)))
    print("Positive->Test: {}".format(len(test_pos)))
    print("Negative->Test: {}".format(len(test_neg)))

    train_pos = parse(train_pos)
    train_neg = parse(train_neg)
    test_pos = parse(test_pos)
    test_neg = parse(test_neg)

    writer(train_pos, train_neg, test_pos, test_neg)


if __name__ == "__main__":
    main()
