import csv
import MeCab
import os


def writer(train_pos, train_neg, test_pos, test_neg):
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
    Parser
    :param sentences: One text
    :return: result of parse
    """
    tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati")
    data = []
    for data_ in sentences:
        parsed_sentence = tagger.parse(data_)
        if parsed_sentence == "\n":
            pass
        else:
            data.append(parsed_sentence.rstrip())
    return data


def fit(data):
    d = []
    for i in data:
        d.append([x for x in i if x.strip()])
    data = d
    comp = []
    comp_ = []
    for text in data:
        for sentences in text:
            if sentences != "\n":
                comp.append(sentences)
        comp_.append(comp)
    comp = [x for x in comp if x.strip()]
    comp__ = []
    for j in comp_:
        comp__.append([x for x in j if x.strip()])
    return d, comp, comp__


def main():
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open("data/tweets_sentiment.10000.train.tsv", newline="") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)

        for row in reader:
            if row[5] == "1":
                train_pos.append(row[3])
            else:
                train_neg.append(row[3])

    with open("data/tweets_sentiment.10000.test.tsv", newline="") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)

        for row in reader:
            if row[5] == "1":
                test_pos.append(row[3])
            else:
                test_neg.append(row[3])
    print(len(train_pos))
    print(len(train_neg))
    print(len(test_pos))
    print(len(test_neg))

    train_pos = parse(train_pos)
    train_neg = parse(train_neg)
    test_pos = parse(test_pos)
    test_neg = parse(test_neg)

    writer(train_pos, train_neg, test_pos, test_neg)


if __name__ == "__main__":
    main()
