import numpy as np
import pandas as pd

from vocabulary import Vocabulary


class DataPod:
    """A class to hold feature data and friends

    Attributes:
        feature_data: a pandas DataFrame with a multi-index by (doc, token), where
            each column represents a feature
        feature_metadata: a boolean mask, True for categorical features
        vocabs: a dictionary with Vocabulary objects for each feature
    """
    def __init__(self, data):
        self.feature_metadata = self.initial_feature_data(data)
        self.feature_metadata = np.bool(1)

    @staticmethod
    def initialise_feature_data(data):
        """Creates the initial dataframe with a multi-index by (doc, token) and one
        all-ones column for bias"""
        # TODO: this can probably be more efficient
        total_len = sum(map(len, data))
        index = np.zeros(total_len)
        offset = 0
        for idx, doc in enumerate(data):
            index[offset: offset + len(doc)] = np.full(idx)
        return pd.DataFrame(np.ones(total_len), index=index)

    def update_feature_data(self, new_features, new_feature_metadata):
        self.feature_data = pd.concat((self.feature_data, new_features), axis=1)
        self.feature_metadata = np.append(self.feature_metadata, new_feature_metadata)


class WordFeatureExtractor:

    # This will learn the vocabs and extract features for
    # ["lower", "word[-3:]", "word[-2:]"]

    def __init__(self):
        # self.vocabs = {name : None for name in ["lower", "word[-3:]", "word[-2:]"]}
        self.vocabs = {name: None for name in ["lower"]}

    def fit_transform(self, tokens, data_pod):
        #TODO: initial version, can be done better
        #TODO: learn vocabulary and add to feature_data.vocabs[self.name]
        vocabs = dict()
        vocabs["lower"] = Vocabulary(use_unks=True, use_start_end=False, use_pad=False)
        # vocab_word3 = Vocabulary(use_unks=True, use_start_end=False, use_pad=False)
        # vocab_word2 = Vocabulary(use_unks=True, use_start_end=False, use_pad=False)
        data_pod.feature_data = pd.concat(
            (data_pod.feature_data, np.zeros(data_pod.feature_data.shape[0], 1)), axis=1
        )
        for doc_idx, doc_tokens in enumerate(tokens):
            for token in doc_tokens:
                vocabs["lower"][token.lower]
                # vocab_word3[token[-3:]], vocab_word2[token[-2:]]
        self.vocabs = vocabs

    def transform(self):




    def
    # def word2features(sent, i):
    #     word = sent[i][0]
    #     postag = sent[i][1]
    #     features = [
    #         'bias',
    #         'word.lower=' + word.lower(),
    #         'word[-3:]=' + word[-3:],
    #         'word[-2:]=' + word[-2:],
    #         'word.isupper=%s' % word.isupper(),
    #         'word.istitle=%s' % word.istitle(),
    #         'word.isdigit=%s' % word.isdigit(),
    #         'postag=' + postag,
    #         'postag[:2]=' + postag[:2],
    #     ]
    #     if i > 0:
    #         word1 = sent[i - 1][0]
    #         postag1 = sent[i - 1][1]
    #         features.extend([
    #             '-1:word.lower=' + word1.lower(),
    #             '-1:word.istitle=%s' % word1.istitle(),
    #             '-1:word.isupper=%s' % word1.isupper(),
    #             '-1:postag=' + postag1,
    #             '-1:postag[:2]=' + postag1[:2],
    #         ])
    #     else:
    #         features.append('BOS')
    #
    #     if i < len(sent) - 1:
    #         word1 = sent[i + 1][0]
    #         postag1 = sent[i + 1][1]
    #         features.extend([
    #             '+1:word.lower=' + word1.lower(),
    #             '+1:word.istitle=%s' % word1.istitle(),
    #             '+1:word.isupper=%s' % word1.isupper(),
    #             '+1:postag=' + postag1,
    #             '+1:postag[:2]=' + postag1[:2],
    #         ])
    #     else:
    #         features.append('EOS')
    #
    #     return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]