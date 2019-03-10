from collections import defaultdict

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def get_tokens_tags_from_sents(sents):
    tokens, tags = [], []
    for sent in sents:
        sent_tokens, _, sent_tags = list(zip(*sent))
        tokens.append(sent_tokens)
        tags.append(sent_tags)
    return tokens, tags


def build_dict(tokens_or_tags, special_tokens):
    # Create a dictionary with default value 0
    tok2idx = defaultdict(lambda: 0)
    ind = 0
    for t in special_tokens:
        tok2idx[t] = ind
        ind += 1
    for sam in tokens_or_tags:
        for t in sam:
            if t not in special_tokens and t not in tok2idx:
                tok2idx[t] = ind
                ind += 1
    return tok2idx, dict((v, k) for k, v in tok2idx.items())


def prepare_sequence(seq, to_ix):
    def to_ix_defaults(t):
        return to_ix.get(t, to_ix.get("<UNK>", to_ix['O']))
    return torch.tensor(np.vectorize(to_ix_defaults)(seq), dtype=torch.long)


def batches_generator(batch_size, tokens, tags, token2idx, tag2idx,
                      shuffle=True, allow_smaller_last_batch=True, seed=8):
    """Generates padded batches of tokens and tags."""
    # TODO: use DataLoader from Pytorch for this
    # tokens is a list of docs, and each docs is a list of tokens
    # SHUFFLE
    n_samples = len(tokens)
    np.random.seed(seed)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    # NUMBER OF BATCHES
    n_batches = n_samples // batch_size
    # and n_samples / batch_size not integer, put the leftovers in last batch
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    # for each batch, get the docs, labels and real lengths and yield them
    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        x, y = [], []
        batch_lengths = torch.zeros(batch_end - batch_start, dtype=torch.int32)
        # x will be a list of lists of indices (one list of indices per doc in this batch)
        for sample_in_batch_index, sample_idx in enumerate(order[batch_start: batch_end]):
            try:
                x.append(prepare_sequence(tokens[sample_idx], token2idx))
                y.append(prepare_sequence(tags[sample_idx], tag2idx))
            except Exception as marc:
                print(marc)
                import pdb; pdb.set_trace()
            batch_lengths[sample_in_batch_index] = len(tags[sample_idx])
        x = pad_sequence(x, batch_first=True, padding_value=token2idx["<PAD>"])
        y = pad_sequence(y, batch_first=True, padding_value=-1)
        batch_lengths, perm_idx = batch_lengths.sort(0, descending=True)
        # yield each batch
        yield x[perm_idx, ...], y[perm_idx, ...], batch_lengths
