import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np

from vocabulary import Vocabulary


class ConLL2002DataSet:
    def __init__(self, data_set):
        self.data = nltk.corpus.conll2002.iob_sents(data_set)

    def get_tokens_tags_from_sents(self):
        tokens, tags = [], []
        for doc in self.data:
            doc_tokens, _, doc_tags = list(zip(*doc))
            tokens.append(doc_tokens)
            tags.append(doc_tags)
        return tokens, tags


class Vectorizer(object):
    def __init__(self, use_start_end=True, use_pad=True):
        self.word_vocab = Vocabulary(
            use_unks=True, use_start_end=use_start_end, use_pad=use_pad
        )
        # start, end, and unk tags will be assigned the same tag: "O"
        self.tag_vocab = Vocabulary(
            use_unks=True, use_start_end=False, unk_token="O", use_pad=use_pad
        )

    def fit(self, docs, tags):
        for doc, doc_tags in zip(docs, tags):
            for word, tag in zip(doc, doc_tags):
                self.word_vocab[word], self.tag_vocab[tag]
        self.word_vocab.freeze()
        self.tag_vocab.freeze()

    @staticmethod
    def map_sequence(vocab, x):
        return torch.tensor(np.vectorize(vocab.__getitem__)(x), dtype=torch.long)

    @staticmethod
    def map_sequence_back(vocab, x):
        return np.vectorize(vocab.get_index)(x)

    def transform(self, all_docs, all_tags):
        transformed_docs, transformed_tags = [], []
        for doc, tags in zip(all_docs, all_tags):
            transformed_docs.append(__class__.map_sequence(self.word_vocab, doc))
            transformed_tags.append(__class__.map_sequence(self.tag_vocab, tags))
        return VectorizedDataset(transformed_docs, transformed_tags)


class VectorizedDataset(Dataset):
    """Dataset where the vectorizer has already been applied"""

    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index], len(self.input[index])


def pad_and_sort_batch(data_loader_batch):
    """
    data_loader_batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    tokens, tags, lengths = tuple(zip(*data_loader_batch))
    x = pad_sequence(tokens, batch_first=True, padding_value=0)
    y = pad_sequence(tags, batch_first=True, padding_value=0)
    lengths, perm_idx = torch.IntTensor(lengths).sort(0, descending=True)
    return x[perm_idx, ...], y[perm_idx, ...], lengths
